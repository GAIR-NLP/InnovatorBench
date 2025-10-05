import subprocess
import os
import signal
import threading
import time
import uuid
import queue
import ptyprocess
import requests
import json
from typing import Optional, Dict, Any, List
from collections import deque
from urllib.parse import urljoin
import re
import shlex


class HTTPTerminalSession:
    """HTTP-based remote terminal session class"""
    
    def __init__(self, session_id: str, computer_ip: str, port: int = 8123, default_shell: str = '/bin/bash', proxy_url: str = None):
        """Initialize HTTP terminal session.

        Args:
            session_id: Unique identifier for the session.
            computer_ip: IP address of the target computer.
            port: Port number for the HTTP connection. Defaults to 8123.
            default_shell: Default shell to use for the session. Defaults to '/bin/bash'.
            proxy_url: Optional proxy URL to route requests through.
        """
        self.session_id = session_id
        self.computer_ip = computer_ip
        self.port = port
        self.proxy_url = proxy_url
        if proxy_url:
            self.base_url = proxy_url  # the url of the proxy
            self.target_host =  f"{computer_ip}:{port}"  # the target host
        else:
            self.base_url = f"http://{computer_ip}:{port}"  # the url of the proxy
            self.target_host = f"{computer_ip}:{port}"   # the target host
        self.is_running = False
        self.current_dir = "/"
        self.lock = threading.Lock()
        self.last_activity = time.time()
        self.default_shell = default_shell
        self.last_health_check = None
        
    def start(self):
        """Start the HTTP terminal session.

        This method initializes and starts a remote terminal session by making HTTP requests
        to create and configure the session.

        Returns:
            bool: True if session started successfully, False otherwise.
        """
        try:
            # create a remote session
            self._make_request('DELETE', f'/api/sessions/{self.session_id}') 
            
            time.sleep(3)
            response = self._make_request('POST', '/api/sessions', {
                'session_id': self.session_id,
                'shell': self.default_shell
            })
            
            
            if response and response.get('success'):
                self.is_running = True
                return True
            else:
                raise Exception(f"Failed to create session: {response.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.is_running = False
            raise Exception(f"Failed to start HTTP session on {self.computer_ip}:{self.port}: {str(e)}")
    
    def _make_request(self, method: str, endpoint: str, data: dict = None, timeout: int = 300) -> dict:
        """send an HTTP request"""
        if self.last_health_check is None or time.time() - self.last_health_check > 3600:
            if endpoint != '/health':
                if not self._check_health():    
                    raise Exception("Health check failed, please check the server is running")
        url = urljoin(self.base_url, endpoint)
        if self.proxy_url:  
            headers = {
                'Content-Type': 'application/json',
                'X-TARGET-HOST': self.target_host  # add the target host header
            }
        else:
            headers = { 
                'Content-Type': 'application/json'
            }
            
        try:
            if method == 'GET':
                # for GET request, use data as query parameters
                response = requests.get(url, headers=headers, params=data, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=timeout)
            else:
                return None

            response.raise_for_status()
            try:
                return response.json() if response.content else {}
            except ValueError:
                return {}
        except requests.RequestException as e:
            print(f"HTTP request failed: {e}")
            return None
    
    def _check_health(self):
        """Check the health status of the HTTP terminal session.

        This method sends a health check request to verify that the remote terminal
        session is responding properly.

        Returns:
            bool: True if health check passes, False otherwise.
        """
        try:
            response = self._make_request('GET', '/health', timeout=30)
            if not response or not response.get('status'):
                raise Exception(f"Failed to check health: {response.get('error', 'Unknown error')}")
        except Exception as e:
            if self.proxy_url:
                print(f"Error checking health: {e}, please check the server proxy: {self.proxy_url} is running")
            else:
                print(f"Error checking health: {e}, please check the server is running")
            return False
        self.last_health_check = time.time()
        return True

    def send_command(self, command: str) -> bool:
        """send a command to the remote session"""
        if not self.is_running:
            return False
            
        try:
            response = self._make_request('POST', f'/api/sessions/{self.session_id}/command', {
                'command': command
            })
            
            if response and response.get('success'):
                self.last_activity = time.time()
                return True
            return False
            
        except Exception as e:
            print(f"Error sending command to session {self.session_id}: {e}")
            return False
    
    def get_output(self, start_lines: int = 50, end_lines: int = None,  since_timestamp: float = None) -> List[Dict]:
        """get the terminal output - directly from the server, not using the local buffer"""
        try:
            # construct query parameters
            params = {}
            if since_timestamp:
                params['since_timestamp'] = since_timestamp
            if start_lines:
                params['start_lines'] = start_lines
            if end_lines:
                params['end_lines'] = end_lines
                
            response = self._make_request('GET', f'/api/sessions/{self.session_id}/output', params)
            
            if response and response.get('success'):
                return response.get('output', [])
            else:
                return []
        except Exception as e:
            print(f"Error getting output from session {self.session_id}: {e}")
            return []
    
    def get_recent_output(self, seconds: int = 10) -> str:
        """get the recent output as a string - directly from the server"""
        cutoff_time = time.time() - seconds
        recent_items = self.get_output(since_timestamp=cutoff_time)
        return ''.join([item.get('content', '') for item in recent_items])
    
    def get_session_timestamp(self) -> float:
        """get the current timestamp"""
        response = self._make_request('GET', '/timestamp')
        if response and isinstance(response, dict) and 'timestamp' in response:
            return response['timestamp']
        # Fallback to local time if request fails or response is not as expected
        return time.time()
    
    def clear_buffer(self):
        """Clear the output buffer on the server.

        This method sends a request to clear the output buffer of the remote terminal session,
        removing all accumulated output data.

        Returns:
            bool: True if buffer cleared successfully, False otherwise.
        """
        try:
            response = self._make_request('DELETE', f'/api/sessions/{self.session_id}/output')
            if response and response.get('success'):
                return True
            else:
                print(f"Failed to clear buffer: {response.get('error', 'Unknown error') if response else 'No response'}")
                return False
        except Exception as e:
            print(f"Error clearing buffer for session {self.session_id}: {e}")
            return False
    
    def is_alive(self) -> bool:
        """check if the session is still alive"""
        if not self.is_running:
            return False
            
        try:
            response = self._make_request('GET', f'/api/sessions/{self.session_id}/status')
            return response and response.get('success') and response.get('is_alive', False)
        except:
            return False
    
    def check_shell_children(self) -> Dict[str, Any]:
        """check if the shell has any child processes running"""
        if not self.is_running:
            return {
                'completed': True,
                'reason': 'session_dead',
                'children_count': 0
            }
            
        try:
            response = self._make_request('GET', f'/api/sessions/{self.session_id}/shell_children')
            if response and response.get('success'):
                return response.get('children_status', {
                    'completed': False,
                    'reason': 'unknown'
                })
            else:
                return {
                    'completed': False,
                    'reason': f'api_error: {response.get("error", "unknown") if response else "no_response"}'
                }
        except Exception as e:
            return {
                'completed': False,
                'reason': f'exception: {str(e)}'
            }

    def check_command_completion(self) -> Dict[str, Any]:
        """check if the current command is executed"""
        if not self.is_running:
            return {
                'completed': True,
                'reason': 'session_not_running'
            }
            
        # use the child process detection method first
        return self.check_shell_children()
    
    def check_waiting_for_input(self, no_output_seconds: int = 20) -> Dict[str, Any]:
        """check if the session is waiting for user input"""
        if not self.is_running:
            return {
                'waiting_for_input': False,
                'reason': 'session_not_running',
                'can_input': False
            }
            
        try:
            response = self._make_request('GET', f'/api/sessions/{self.session_id}/check_input', {
                'no_output_seconds': no_output_seconds
            })
            
            if response and response.get('success'):
                return response.get('input_status', {
                    'waiting_for_input': False,
                    'reason': 'unknown',
                    'can_input': False
                })
            else:
                return {
                    'waiting_for_input': False,
                    'reason': f'api_error: {response.get("error", "unknown") if response else "no_response"}',
                    'can_input': False
                }
        except Exception as e:
            return {
                'waiting_for_input': False,
                'reason': f'exception: {str(e)}',
                'can_input': False
            }
    
    def send_input(self, input_text: str) -> Dict[str, Any]:
        """send input to the remote session"""
        if not self.is_running:
            return {
                'success': False,
                'message': 'Session is not running'
            }
            
        try:
            response = self._make_request('POST', f'/api/sessions/{self.session_id}/input', {
                'input': input_text
            })
            
            if response and response.get('success'):
                self.last_activity = time.time()
                return response
            else:
                return {
                    'success': False,
                    'message': response.get('message', 'Unknown error') if response else 'No response'
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f"Error sending input to session {self.session_id}: {str(e)}"
            }

    def kill_processes(self, force: bool = False) -> Dict[str, Any]:
        """
        kill all the child processes running in the remote session
        
        Args:
            force: whether to use SIGKILL to force kill (default use SIGTERM）
            
        Returns:
            the execution result dictionary
        """
        if not self.is_running:
            return {
                'success': True,
                'message': f"Session {self.session_id} is not running",
                'killed_processes': []
            }
        
        try:
            response = self._make_request('POST', f'/api/sessions/{self.session_id}/kill_processes', {
                'force': force
            })
            
            if response and response.get('success'):
                return response
            else:
                return {
                    'success': False,
                    'error': response.get('error', 'Unknown error') if response else 'No response'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Error killing processes in session {self.session_id}: {str(e)}"
            }
    
    def terminate(self):
        """terminate the session"""
        self.is_running = False
        
        try:
            self._make_request('DELETE', f'/api/sessions/{self.session_id}')
        except:
            pass  # ignore the error when deleting

class TerminalSession:
    """a single terminal session class, supporting local connection"""
    
    def __init__(self, session_id: str, shell: str = '/bin/bash', buffer_size: int = 10000):
        """Initialize local terminal session.

        Args:
            session_id: Unique identifier for the session.
            shell: Shell to use for the session. Defaults to '/bin/bash'.
            buffer_size: Maximum number of lines to keep in output buffer. Defaults to 10000.
        """
        self.session_id = session_id
        self.shell = shell
        self.process = None
        self.output_buffer = deque(maxlen=buffer_size)  # the output buffer, max save 10000 lines
        self.is_running = False
        self.current_dir = "/"
        self.lock = threading.Lock()
        self.output_thread = None
        self.command_queue = queue.Queue()
        self.last_activity = time.time()
        self.created_at = time.time()  # add the creation time
        # add command completion detection related attributes
        self.command_completion_marker = None
        self.last_command_timestamp = None
        self.command_in_progress = False
        
    def start(self):
        """Start the local terminal session.

        This method initializes and starts a local terminal session by spawning
        a pseudo-terminal process using ptyprocess.

        Returns:
            bool: True if session started successfully, False otherwise.
        """
        try:
            # create a pseudo terminal process
            self.process = ptyprocess.PtyProcess.spawn([self.shell])
            self.process.setwinsize(100, 300)
            self.is_running = True
            
            # start the output reading thread
            self.output_thread = threading.Thread(target=self._read_output, daemon=True)
            self.output_thread.start()
            
            # wait for the shell to start
            time.sleep(0.1)
            
            return True
        except Exception as e:
            self.is_running = False
            raise Exception(f"Failed to start local terminal session: {str(e)}")
    
    def _read_output(self):
        """Read terminal output in the background thread.

        This method runs in a separate thread and continuously reads output from the
        pseudo-terminal process, storing it in the output buffer with timestamps.

        The method processes the output line by line and stores each line with its
        timestamp in the output buffer for later retrieval.
        """
        while self.is_running and self.process and self.process.isalive():
            try:
                # PtyProcess.read() does not support the timeout parameter, use the non-blocking read
                output = self.process.read()
                if output:
                    with self.lock:
                        # split the output by lines and add to the buffer
                        lines = output.decode('utf-8', errors='replace').splitlines(True)
                        for line in lines:
                            self.output_buffer.append({
                                'timestamp': time.time(),
                                'content': line
                            })
                        self.last_activity = time.time()
                
                # reduce the delay, improve the responsiveness
                time.sleep(0.05)
                
            except (EOFError, OSError):
                # the process ends or the connection is disconnected
                break
            except Exception as e:
                print(f"Error reading output from session {self.session_id}: {e}")
                break

    def get_session_timestamp(self) -> float:
        """get the current timestamp"""
        return time.time()
    
     
    def send_command(self, command: str) -> bool:
        """send a command to the terminal"""
        if not self.is_running or not self.process or not self.process.isalive():
            return False
        
        try:
            # mark the command as started
            with self.lock:
                self.command_in_progress = True
                self.last_command_timestamp = time.time()
                # generate a unique completion marker
                self.command_completion_marker = f"CMD_COMPLETE_{uuid.uuid4().hex[:8]}"
            
            # ensure the terminal is ready
            time.sleep(0.05)
            
            # for very long commands (more than 500 chars), use the chunked sending
            if len(command) > 500:
                return self._send_long_command(command)
            else:
                # short commands are sent directly
                self.process.write((command + '\n').encode('utf-8'))
            self.process.flush()
            self.last_activity = time.time()
            return True
        except Exception as e:
            print(f"Error sending command to session {self.session_id}: {e}")
            return False
    
    def _send_long_command(self, command: str) -> bool:
        """send the long command in chunks, avoid the buffer overflow"""
        try:
            command_bytes = (command + '\n').encode('utf-8')
            chunk_size = 512  # 512 bytes chunk size, more conservative
            
            for i in range(0, len(command_bytes), chunk_size):
                chunk = command_bytes[i:i + chunk_size]
                self.process.write(chunk)
                # short delay, ensure the data is processed
                time.sleep(0.02)
            
            return True
        except Exception as e:
            print(f"Error sending long command to session {self.session_id}: {e}")
            return False
    
    def get_output(self, start_lines: int = 50, end_lines: int = None, since_timestamp: float = None) -> List[Dict]:
        """get the terminal output"""
        with self.lock:
            if since_timestamp:
                # get the output after the specified timestamp
                result = [item for item in self.output_buffer 
                         if item['timestamp'] > since_timestamp]
            else:
                # get the recent specified lines
                result = list(self.output_buffer)[-start_lines:] if start_lines > 0 else list(self.output_buffer)
                if end_lines and len(result) > end_lines:
                    result = result[:-end_lines]
            
            return result
    
    def get_recent_output(self, seconds: int = 10) -> str:
        """get the recent output as a string"""
        cutoff_time = time.time() - seconds
        recent_items = self.get_output(since_timestamp=cutoff_time)
        return ''.join([item['content'] for item in recent_items])
    
    def clear_buffer(self) -> bool:
        """Clear the output buffer.

        This method clears all accumulated output data from the terminal session's
        output buffer, removing all stored lines.

        Returns:
            bool: True if buffer cleared successfully, False otherwise.
        """
        try:
            with self.lock:
                self.output_buffer.clear()
            return True
        except Exception as e:
            print(f"Error clearing buffer for session {self.session_id}: {e}")
            return False

    
    def _get_recent_output_content(self, seconds: int) -> str:
        """get the recent output content"""
        cutoff_time = time.time() - seconds
        with self.lock:
            recent_items = [item for item in self.output_buffer 
                           if item['timestamp'] > cutoff_time]
            return ''.join([item['content'] for item in recent_items])
    
    def is_alive(self) -> bool:
        """check if the session is still alive"""
        return self.is_running and self.process and self.process.isalive()

    def terminate(self):
        """Terminate the HTTP terminal session.

        This method closes the remote terminal session and cleans up any associated resources.
        It sends a DELETE request to terminate the session on the server.

        Returns:
            bool: True if session terminated successfully, False otherwise.
        """
        self.is_running = False
        if self.process and self.process.isalive():
            try:
                self.process.kill(signal.SIGTERM)
                self.process.wait(timeout=5)
            except:
                self.process.kill(signal.SIGKILL)
        
        if self.output_thread and self.output_thread.is_alive():
            self.output_thread.join(timeout=2)

    def check_shell_children(self) -> Dict[str, Any]:
        """check if the shell has any child processes running"""
        if not self.is_alive():
            return {
                'completed': True,
                'reason': 'the session is dead.',
                'children_count': 0
            }
        
        # if there is no command in progress, return the completed status
        if not self.command_in_progress:
            return {
                'completed': True,
                'reason': 'there is no command in progress.',
                'children_count': 0
            }
        
        try:
            # get the shell process PID
            shell_pid = self.process.pid
            
            # first check if the shell process itself still exists
            import subprocess
            shell_check = subprocess.run(
                ['ps', '-p', str(shell_pid)],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if shell_check.returncode != 0:
                # the shell process does not exist, it means it has exited, the command must be completed
                with self.lock:
                    self.command_in_progress = False
                    self.command_completion_marker = None
                
                return {
                    'completed': True,
                    'reason': 'the shell process has exited.',
                    'children_count': 0,
                    'children_info': []
                }
            
            # the shell process exists, continue to check the child processes
            # use the recursive method to get all the child processes (including the grandchild processes)
            def get_all_children(parent_pid):
                """recursively get all the child processes"""
                children = []
                try:
                    result = subprocess.run(
                        ['ps', '--ppid', str(parent_pid), '-o', 'pid,ppid,stat,cmd', '--no-headers'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if result.returncode == 0 and result.stdout.strip():
                        lines = result.stdout.strip().split('\n')
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            
                            parts = line.split(None, 3)
                            if len(parts) < 4:
                                continue
                            
                            pid, ppid, stat, cmd = parts
                            children.append((pid, ppid, stat, cmd))
                            
                            # recursively check the child processes of this process
                            grandchildren = get_all_children(pid)
                            children.extend(grandchildren)
                            
                except Exception:
                    pass
                
                return children
            
            # get all the child processes
            all_children = get_all_children(shell_pid)
            
            # more strict filtering: only focus on the actual command execution process
            actual_children = []
            children_info_output = ""
            for pid, ppid, stat, cmd in all_children:
                
                # use more precise matching conditions to filter out the irrelevant processes
                should_exclude = False
                
                # Filter ps command itself
                if cmd.startswith('ps ') and '--ppid' in cmd:
                    should_exclude = True
                
                # Filter pure bash processes (empty bash without executing scripts)
                elif cmd.strip() in ['/bin/bash', 'bash', '-bash']:
                    should_exclude = True
                
                # Filter kernel threads (process names starting and ending with brackets)
                elif cmd.startswith('[') and cmd.endswith(']') and ' ' not in cmd.strip('[]'):
                    should_exclude = True
                
                if should_exclude:
                    continue
                
                # Check process status, filter out zombie and stopped processes
                if 'Z' in stat or 'T' in stat:  # Z=zombie, T=stopped
                    continue
                
                actual_children.append(f"{pid} {stat} {cmd}")
                children_info_output += f"- Process {pid} is in {stat} status, running command `{cmd}`.\n"
            
            children_count = len(actual_children)
            
            # If there are no real child processes, the command may have completed
            if children_count == 0:
                with self.lock:
                    self.command_in_progress = False
                    self.command_completion_marker = None
                
                return {
                    'completed': True,
                    'reason': 'there is no active child process.',
                    'children_count': 0,
                    'children_info': []
                }
            else:
                return {
                    'completed': False,
                    'reason': f'the session has the following {children_count} active child {"processes" if children_count > 1 else "process"}:\n{children_info_output}',
                    'children_count': children_count,
                    'children_info': actual_children
                }
        except Exception as e:
            return {
                'completed': False,
                'reason': f'error: {str(e)}',
                'children_count': 0,
                'children_info': []
            }
    
    def check_command_completion(self) -> Dict[str, Any]:
        """Check if the current command has completed execution - prioritize child process detection"""
        if not self.command_in_progress:
            return {
                'completed': True,
                'reason': 'no_command_in_progress'
            }
        
        if not self.is_alive():
            return {
                'completed': True,
                'reason': 'session_dead'
            }
        
        # Prioritize child process detection method
        return self.check_shell_children()
    
    def check_waiting_for_input(self, no_output_seconds: int = 20) -> Dict[str, Any]:
        """Check if the session is waiting for user input"""
        if not self.is_alive():
            return {
                'waiting_for_input': False,
                'reason': 'the session is dead.',
                'can_input': False
            }
        
        if not self.command_in_progress:
            return {
                'waiting_for_input': False,
                'reason': 'there is no command in progress',
                'can_input': False
            }
        
        # Check if there are child processes running
        children_status = self.check_shell_children()
        if children_status.get('completed', True):
            return {
                'waiting_for_input': False,
                'reason': 'the command has completed.',
                'can_input': False
            }
        
        # Check if there has been recent output
        recent_output_time = 0
        with self.lock:
            if self.output_buffer:
                # Get the timestamp of the most recent output
                recent_output_time = self.output_buffer[-1]['timestamp']
        
        current_time = time.time()
        seconds_since_output = current_time - recent_output_time
        
        # If there has been no output for the specified time, assume it's waiting for input
        if seconds_since_output >= no_output_seconds:
            return {
                'waiting_for_input': True,
                'reason': f'there is no output for {seconds_since_output:.1f} seconds',
                'can_input': True,
                'seconds_since_output': seconds_since_output,
                'children_count': children_status.get('children_count', 0),
                'children_info': children_status.get('children_info', [])
            }
        else:
            return {
                'waiting_for_input': False,
                'reason': f'the recent output is {seconds_since_output:.1f} seconds ago',
                'can_input': False,
                'seconds_since_output': seconds_since_output
            }
    
    def send_input(self, input_text: str) -> Dict[str, Any]:
        """Send input to terminal"""
        if not input_text:
            return {
                'success': False,
                'message': 'Input text cannot be empty'
            }
        
        if not self.is_alive():
            return {
                'success': False,
                'message': 'Session is not alive'
            }
        
        # Check if input conditions are met
        input_check = self.check_waiting_for_input()
        if not input_check.get('can_input', False):
            return {
                'success': False,
                'message': f"Session is not waiting for input: {input_check.get('reason', 'unknown')}",
                'input_check': input_check
            }
        
        try:
            # Send input (automatically add newline)
            if not input_text.endswith('\n'):
                input_text += '\n'
            
            self.process.write(input_text.encode('utf-8'))
            self.last_activity = time.time()
            
            return {
                'success': True,
                'message': 'Input sent successfully',
                'input_length': len(input_text)
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Failed to send input: {str(e)}"
            }

    def kill_processes(self, force: bool = False) -> Dict[str, Any]:
        """
        Kill all child processes running in the session

        Args:
            force: Whether to use SIGKILL to force kill (default uses SIGTERM first)

        Returns:
            Execution result dictionary
        """
        if not self.is_alive():
            return {
                'success': True,
                'message': f"Session {self.session_id} is not alive",
                'killed_processes': []
            }
        
        try:
            # Get active child processes
            children_info = self.check_shell_children()
            
            if children_info.get('completed', True):
                return {
                    'success': True,
                    'message': f"No active processes to kill in session {self.session_id}",
                    'killed_processes': []
                }
            
            children_list = children_info.get('children_info', [])
            if not children_list:
                return {
                    'success': True,
                    'message': f"No child processes found in session {self.session_id}",
                    'killed_processes': []
                }
            
            killed_processes = []
            failed_kills = []

            # Parse and kill each child process
            for child_info in children_list:
                try:
                    # Child process info format: "PID STAT CMD"
                    parts = child_info.split(None, 2)
                    if len(parts) < 3:
                        continue
                    
                    pid = int(parts[0])
                    stat = parts[1]
                    cmd = parts[2]

                    # Use kill command to kill the process
                    import subprocess
                    kill_signal = 'KILL' if force else 'TERM'
                    
                    kill_result = subprocess.run(
                        ['kill', f'-{kill_signal}', str(pid)],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if kill_result.returncode == 0:
                        killed_processes.append({
                            'pid': pid,
                            'cmd': cmd,
                            'signal': kill_signal,
                            'status': 'killed'
                        })
                    else:
                        failed_kills.append({
                            'pid': pid,
                            'cmd': cmd,
                            'error': kill_result.stderr.strip(),
                            'status': 'failed'
                        })
                        
                except (ValueError, subprocess.TimeoutExpired) as e:
                    failed_kills.append({
                        'info': child_info,
                        'error': str(e),
                        'status': 'error'
                    })

            # If using TERM signal, wait a bit and check if there are still processes that need force killing
            if not force and killed_processes:
                time.sleep(1)  # Wait for graceful process exit

                # Check again if there are still child processes
                remaining_check = self.check_shell_children()
                if not remaining_check.get('completed', True):
                    remaining_children = remaining_check.get('children_info', [])
                    if remaining_children:
                        # Some processes didn't exit, use SIGKILL to force kill
                        for child_info in remaining_children:
                            try:
                                parts = child_info.split(None, 2)
                                if len(parts) >= 3:
                                    pid = int(parts[0])
                                    cmd = parts[2]
                                    
                                    subprocess.run(['kill', '-KILL', str(pid)], 
                                                 capture_output=True, timeout=5)
                                    killed_processes.append({
                                        'pid': pid,
                                        'cmd': cmd,
                                        'signal': 'KILL',
                                        'status': 'force_killed'
                                    })
                            except:
                                pass

            # Mark command execution as completed
            with self.lock:
                self.command_in_progress = False
                self.command_completion_marker = None
            
            result = {
                'success': True,
                'message': f"Killed {len(killed_processes)} processes in session {self.session_id}",
                'killed_processes': killed_processes,
                'signal_used': 'KILL' if force else 'TERM'
            }
            
            if failed_kills:
                result['failed_kills'] = failed_kills
                result['message'] += f", {len(failed_kills)} failed"
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Error killing processes: {str(e)}"
            }


class CmdOperations:
    """Enhanced command execution operations class, supporting local and HTTP remote connections"""
    
    def __init__(self, default_shell: str = '/bin/bash', default_http_port: int = 8123, proxy_url: str = None, env_vars: Dict[str, str] = None):
        self.default_shell = default_shell
        self.default_http_port = default_http_port
        
        # Sessions are now managed grouped by computer_ip
        self.sessions: Dict[str, Dict[str, Any]] = {}  # {computer_ip: {session_id: session}}
        self.default_sessions: Dict[str, str] = {}  # {computer_ip: default_session_id}
        self.lock = threading.Lock()
        self.proxy_url = proxy_url
        self.env_vars = env_vars
        
        # Create local default session
        self.create_session(computer_ip='localhost')
        self.MAX_OUTPUT_LENGTH = 30000

    def _output_error(self, error_msg: str) -> Dict[str, Any]:
        """Output error message"""
        return {"success": False, "error": error_msg}

    def _get_session_key(self, computer_ip: str, session_id: str) -> str:
        """Generate unique key for session"""
        return f"{computer_ip}:{session_id}"

    def _is_local_ip(self, computer_ip: str) -> bool:
        """Determine if it's a local IP"""
        return computer_ip in ['localhost', '127.0.0.1', '::1'] or computer_ip == self._get_local_ip()

    def _get_local_ip(self) -> str:
        """Get local machine IP"""
        import socket
        try:
            # Connect to an external address to get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except:
            return ips
        
    def _set_environment(self, computer_ip, session_id, http_port, wait_for_completion, use_proxy):
        if self.env_vars:
            cmd = f"export {' '.join([f'{key}={value}' for key, value in self.env_vars.items()])}"
            self.run_command(cmd, computer_ip, session_id, http_port, wait_for_completion=True, use_proxy=use_proxy)
            # session.send_command(cmd)
    
    def _validate_rm_command(self, command: str) -> Dict[str, Any]:
        """
        Validate whether the rm command in the given string is safe:
        1. Detect all rm commands in the command chain (including those connected by &&, ||, ;, |, etc.)
        2. Only allow deletion of absolute paths
        3. Only allow deletion of content under /workspace
        
        Returns:
            Dict with 'valid' boolean and 'error' message if invalid
        """
        
        # Check if the command contains rm
        if 'rm' not in command:
            return {'valid': True, 'error': None}
        
        # Split the command by common separators
        separators = r'(\s*(?:&&|\|\||;)\s*)'
        parts = re.split(separators, command)
        
        # Extract actual command parts (skip separators)
        commands = []
        for i, part in enumerate(parts):
            if i % 2 == 0:  # even index -> command
                stripped = part.strip()
                if stripped:
                    commands.append(stripped)
        
        # If no separator, treat as single command
        if len(commands) <= 1:
            commands = [command.strip()]
        
        # Check each sub-command
        for sub_command in commands:
            if not sub_command.strip():
                continue
                
            # Skip if no rm
            if 'rm' not in sub_command:
                continue
            
            # Parse command tokens
            try:
                tokens = shlex.split(sub_command)
            except ValueError as e:
                return {
                    'valid': False,
                    'error': f'Failed to parse command "{sub_command}": {str(e)}'
                }
            
            if not tokens:
                continue
            
            # Check if rm exists
            rm_found = False
            rm_command_start = -1
            
            for i, token in enumerate(tokens):
                if token == 'rm':
                    rm_found = True
                    rm_command_start = i
                    break
            
            if not rm_found:
                if re.search(r'\brm\s+/', sub_command):
                    return {
                        'valid': False,
                        'error': f'Potential rm bypass attempt detected: "{sub_command}"'
                    }
                continue
            
            # Extract rm arguments
            rm_tokens = tokens[rm_command_start:]
            paths = []
            for token in rm_tokens[1:]:
                if not token.startswith('-'):  # skip options
                    paths.append(token)
            
            if not paths:
                return {
                    'valid': False,
                    'error': f'rm command must specify a target path: "{sub_command}"'
                }
            
            # Validate each path
            for path in paths:
                if not path.startswith('/'):
                    return {
                        'valid': False,
                        'error': f'Path "{path}" is not an absolute path. For safety, rm commands must use absolute paths. (In command: "{sub_command}")'
                    }
                
                normalized_path = os.path.normpath(path)
                
                if not normalized_path.startswith('/workspace'):
                    return {
                        'valid': False,
                        'error': f'Path "{path}" is not under /workspace. For safety, rm commands can only delete files inside /workspace. (In command: "{sub_command}")'
                    }
                
                if normalized_path == '/workspace':
                    return {
                        'valid': False,
                        'error': f'Deletion of /workspace directory itself is not allowed. (In command: "{sub_command}")'
                    }
        
        return {'valid': True, 'error': None}
    
    def _validate_kill_commands(self, command: str) -> Dict[str, Any]:
        """
        Validate whether kill/pkill commands in the given string are allowed:
        1. Detect all kill/pkill commands in the command chain (including those connected by &&, ||, ;, |, etc.)
        2. Completely forbidden, must use kill_session_processes instead
        
        Returns:
            Dict with 'valid' boolean and 'error' message if invalid
        """
        
        # Check if the command contains kill or pkill
        if 'kill' not in command and 'pkill' not in command and 'killall' not in command:
            return {'valid': True, 'error': None}
        
        # Split the command by common separators
        separators = r'(\s*(?:&&|\|\||;)\s*)'
        parts = re.split(separators, command)
        
        # Extract actual command parts (skip separators)
        commands = []
        for i, part in enumerate(parts):
            if i % 2 == 0:  # even index -> command
                stripped = part.strip()
                if stripped:
                    commands.append(stripped)
        
        # If no separator, treat as single command
        if len(commands) <= 1:
            commands = [command.strip()]
        
        # Check each sub-command
        for sub_command in commands:
            if not sub_command.strip():
                continue
            
            # Parse command tokens
            try:
                tokens = shlex.split(sub_command)
            except ValueError as e:
                return {
                    'valid': False,
                    'error': f'Failed to parse command "{sub_command}": {str(e)}'
                }
            
            if not tokens:
                continue
            
            # Check for kill/pkill commands
            for kill_cmd in ['kill', 'pkill', 'killall']:
                cmd_found = False
                
                for i, token in enumerate(tokens):
                    if token == kill_cmd:
                        cmd_found = True
                        break
                
                if not cmd_found:
                    # Check for potential bypass attempts
                    if re.search(rf'\b{kill_cmd}\s+/', sub_command):
                        return {
                            'valid': False,
                            'error': f'Potential {kill_cmd} bypass attempt detected: "{sub_command}"'
                        }
                    continue
                
                # kill/pkill commands are completely forbidden
                return {
                    'valid': False,
                    'error': f'{kill_cmd} command is not allowed. Please use `kill_session_processes` to kill processes in the corresponding session instead.'
                }
        
        return {'valid': True, 'error': None}
    
    def create_session(self, computer_ip: str = 'localhost', session_id: str = None,
                      http_port: int = None, use_proxy: bool = True) -> Dict[str, Any]:
        """Create a new terminal session on the computer specified by `computer_ip`.
        
        This function initializes connectivity via `http_port` and `use_proxy`. Use `use_proxy=False` for `cpu`/`localhost_cpu` 
        machines and `use_proxy=True` for `gpu` machines.
        
        Args:
            computer_ip: str: The IP address of the computer. Default is 'localhost'.
            session_id: str: Unique identifier of the target session. If absent, a new session is created and a new 
                            `session_id` is assigned on the host `computer_ip`. Default is None.
            http_port: int: The HTTP port to use to connect to the session. Default is None.
            use_proxy: bool: Whether to use a proxy for connecting to the session. Set `use_proxy=False` for `cpu` and 
                            `localhost_cpu` computers, and set `use_proxy=True` for `gpu` computers. Must align with 
                            your network topology or the connection will fail. Default is True.
            
        Returns:
            Dict[str, Any]: Dictionary containing session creation status and information.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]

        # Ensure sessions dictionary for computer_ip exists
        if computer_ip not in self.sessions:
            self.sessions[computer_ip] = {}
        
        if session_id in self.sessions[computer_ip]:
            return {
                'success': True,
                'message': f"Session {session_id} already exists on {computer_ip}",
                'computer_ip': computer_ip,
                'session_id': session_id,
            }
            # raise ValueError(f"Session {session_id} already exists on {computer_ip}")
        
        # if True:
        try:
            with self.lock:
                if self._is_local_ip(computer_ip):
                    # Create local session
                    session = TerminalSession(session_id=session_id, shell=self.default_shell)
                else:
                    # Create HTTP remote session
                    port = http_port or self.default_http_port
                    if self.proxy_url and use_proxy:
                        session = HTTPTerminalSession(
                            session_id=session_id, 
                            computer_ip=computer_ip, 
                            port=port,
                            default_shell=self.default_shell,
                            proxy_url=self.proxy_url
                            )
                    else:
                        session = HTTPTerminalSession(
                            session_id=session_id, 
                            computer_ip=computer_ip, 
                            port=port,
                            default_shell=self.default_shell
                        )
                
                session.start()

                self.sessions[computer_ip][session_id] = session

                # If this is the first session on this machine, set it as the default session
                if computer_ip not in self.default_sessions:
                    self.default_sessions[computer_ip] = session_id
            self._set_environment(computer_ip, session_id, http_port, wait_for_completion=True, use_proxy=use_proxy)
            self.run_command("cd /workspace", computer_ip, session_id, http_port, wait_for_completion=True, use_proxy=use_proxy)
            return {
                'success': True,
                'message': f"Session {session_id} created successfully on {computer_ip}",
                'computer_ip': computer_ip,
                'session_id': session_id,
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"Failed to create session {session_id} on {computer_ip}: {str(e)}",
                'computer_ip': computer_ip,
                'session_id': session_id,
            }

    def list_sessions(self, computer_ip: str = None) -> Dict[str, Any]:
        """List all existing sessions.
        
        Key '<computer_ip>:<session_id>' on the output refers to the session <session_id> on <computer_ip>.
        
        Args:
            computer_ip: str: The IP address of the computer. If None, lists sessions on all machines. Default is None.
            
        Returns:
            Dict[str, Any]: Dictionary containing information about all active sessions.
        """
        result = {}
        with self.lock:
            if computer_ip:
                # Only list sessions for specified machine
                if computer_ip in self.sessions:
                    for sid, session in self.sessions[computer_ip].items():
                        key = self._get_session_key(computer_ip, sid)
                        result[key] = {
                            'computer_ip': computer_ip,
                            'session_id': sid,
                            'is_alive': session.is_alive(),
                            'last_activity': session.last_activity,
                            # 'current_dir': getattr(session, 'current_dir', '/'),
                            'is_default': sid == self.default_sessions.get(computer_ip),
                            'connection_type': 'local' if self._is_local_ip(computer_ip) else 'http',
                            "is_idle": self.session_idle(computer_ip,sid)['is_idle'],
                        }
            else:
                # List sessions for all machines
                for machine_ip, machine_sessions in self.sessions.items():
                    for sid, session in machine_sessions.items():
                        key = self._get_session_key(machine_ip, sid)
                        result[key] = {
                            'computer_ip': machine_ip,
                            'session_id': sid,
                            'is_alive': session.is_alive(),
                            'last_activity': session.last_activity,
                            # 'current_dir': getattr(session, 'current_dir', '/'),
                            'is_default': sid == self.default_sessions.get(machine_ip),
                            'connection_type': 'local' if self._is_local_ip(machine_ip) else 'http',
                            "is_idle": self.session_idle(machine_ip,sid)['is_idle'],
                        }
        return {
            'success': True,
            'message': f"Listed {len(result)} sessions on {computer_ip if computer_ip else 'all machines'}",
            'output': result
        }
    
    
    def run_command(
        self,
        command: str,
        computer_ip: str = 'localhost',
        session_id: str = None,
        http_port: int = None,
        wait_for_completion: bool = False,
        use_proxy: bool = True
    ) -> Dict[str, Any]:
        """Execute a single bash command in the session identified by `session_id`.
        
        If the session does not exist, it will be created and bound to the target host (determined by `computer_ip`) 
        and will be connected via `http_port` and `use_proxy`. Only one command may run concurrently per session.
        
        Args:
            command: str: Shell (bash) command to execute in the target session's working directory and environment.
            computer_ip: str: The IP address of the computer. Default is 'localhost'.
            session_id: str: Unique identifier of the target session. If absent, a new session is created on the host 
                             determined by `computer_ip`. Default is None.
            http_port: int: The HTTP port to use to connect to the session. Default is None.
            wait_for_completion: bool: Whether to block until the command finishes:
                                      - True: block up to 10 seconds; on timeout the command process is killed.
                                      - False: return immediately and let the command run in the background.
                                      Default is False.
            use_proxy: bool: Whether to use a proxy for connecting to the session. Set `use_proxy=False` for `cpu` and 
                             `localhost_cpu` computers, and set `use_proxy=True` for `gpu` computers. Default is True.
        
        Returns:
            Dict[str, Any]: Dictionary containing command execution results and status.
        """
        if not command or not command.strip():
            raise ValueError('Command cannot be empty')

        # Ensure target machine has available session
        if computer_ip not in self.sessions or not self.sessions[computer_ip] or (session_id is not None and session_id not in self.sessions[computer_ip]):
            # Auto-create session
            try:
                if session_id:
                    create_result = self.create_session(
                        computer_ip=computer_ip, 
                        http_port=http_port,
                        use_proxy=use_proxy,
                        session_id=session_id
                    )
                else:
                    create_result = self.create_session(
                        computer_ip=computer_ip, 
                        http_port=http_port,
                        use_proxy=use_proxy
                    )
                print(f"Auto-created session {create_result['session_id']} on {computer_ip}")
            except Exception as e:
                return {
                    'success': False,
                    'message': f'Failed to create session on {computer_ip}: {str(e)}',
                    'computer_ip': computer_ip,
                    'session_id': None,
                    'command': command,
                    'async': None,
                    'execution_time': 0,
                    'timestamp': time.time(),
                    'output': None,
                }
        
        # Determine which session to use
        target_session_id = session_id or self.default_sessions.get(computer_ip)
        if not target_session_id or target_session_id not in self.sessions[computer_ip]:
            return {
                'success': False,
                'message': f'No valid session found on {computer_ip}',
                'computer_ip': computer_ip,
                'session_id': target_session_id,
                'command': command,
                'async': None,
                'execution_time': 0,
                'timestamp': time.time(),
                'output': None,
            }
        
        session = self.sessions[computer_ip][target_session_id]
        if not session.is_alive():
            # Try to recreate session
            try:
                self.close_session(computer_ip, target_session_id)
                create_result = self.create_session(
                    computer_ip=computer_ip, 
                    http_port=http_port,
                    use_proxy=use_proxy
                )
                if not create_result.get('success'):
                    raise Exception('Failed to recreate session.')
                
                new_session_id = create_result['session_id']
                session = self.sessions[computer_ip][new_session_id]
                target_session_id = new_session_id
                print(f"Recreated session {new_session_id} on {computer_ip}")
            except Exception as e:
                return {
                    'success': False,
                    'message': f'Session {target_session_id} on {computer_ip} is not alive and failed to recreate: {str(e)}',
                    'computer_ip': computer_ip,
                    'session_id': target_session_id,
                    'command': command,
                    'async': None,
                    'execution_time': 0,
                    'timestamp': time.time(),
                    'output': None,
                }
        
        # Ensure terminal has no other processes running
        if not session.check_shell_children()['completed']:
            return {
                'success': False,
                'message': 'Current session is busy, there are three options for you to choose:\n(1) Wait last command to finish\n(2) Kill this process, but be careful\n(3) Create a new session to run this command\n(4) Input content using `input_in_session` tool if the session requires input',
                'computer_ip': computer_ip,
                'session_id': target_session_id,
                'command': command,
                'async': None,
                'execution_time': 0,
                'timestamp': time.time(),
                'output': None,
            }

        if "rm" in command:
            # Validate rm command safety
            validation = self._validate_rm_command(command)
            if not validation['valid']:
                return {
                    'success': False,
                    'message': f'rm command is not allowed: {validation["error"]}',
                    'computer_ip': computer_ip,
                    'session_id': target_session_id,
                    'command': command,
                    'async': None,
                    'execution_time': 0,
                    'timestamp': time.time(),
                    'output': None,
                }
        
        if "kill" in command:
            # Validate kill/pkill commands
            validation = self._validate_kill_commands(command)
            if not validation['valid']:
                return {
                    'success': False,
                    'message': f'kill/pkill command is not allowed: {validation["error"]}',
                    'computer_ip': computer_ip,
                    'session_id': target_session_id,
                    'command': command,
                    'async': None,
                    'execution_time': 0,
                    'timestamp': time.time(),
                    'output': None,
                }
                
        if "nohup" in command:
            return {
                'success': False,
                'message': f'nohup command is not allowed',
                'computer_ip': computer_ip,
                'session_id': target_session_id,
                'command': command,
                'async': None,
                'execution_time': 0,
                'timestamp': time.time(),
                'output': None,
            }

        # Record timestamp before sending command
        start_time = time.time()
        session_timestamp = session.get_session_timestamp()
        # Send command
        success = session.send_command(command)
        if not success:
            return {
                'success': False,
                'message': 'Failed to send command to session',
                'computer_ip': computer_ip,
                'session_id': target_session_id,
                'command': command,
                'async': None,
                'execution_time': 0,
                'timestamp': time.time(),
                'output': None,
            }
        
        if not wait_for_completion:
            # Async execution, return immediately
            return {
                'success': True,
                'message': f'Command sent to session {target_session_id} on {computer_ip}',
                'computer_ip': computer_ip,
                'session_id': target_session_id,
                'command': command,
                'async': True,
                'execution_time': time.time() - start_time,
                "timestamp": time.time(),
                'output': None,
            }
        
        # Wait for command completion
        time.sleep(2)
        killed = False
        while not session.check_command_completion()['completed']:
            if time.time() - start_time >= 10.5:
                self.kill_session_processes(computer_ip, target_session_id, force=True)
                kill_times = 0
                while kill_times < 30:
                    output = self.get_session_output(computer_ip, target_session_id, since_timestamp=session_timestamp)
                    print(output['output'])
                    if "Killed" in output['output']:
                        break
                    time.sleep(1)
                    kill_times += 1
                killed = True
                break
            time.sleep(1)

        time.sleep(1) # Wait 1 second to give session time to process
        
        
        output = self.get_session_output(computer_ip, target_session_id, since_timestamp=session_timestamp)
        
        if killed:
            message = f'Command `{command}` on {computer_ip} executed in {time.time() - start_time} seconds, which surpasses max execution time in `wait_for_completion` mode. Hence, it is killed. If you want to execute commands that will be run for more than 10 seconds, please set `wait_for_completion=False` and use `get_session_output` to get the command content from the session.'
        else:
            message = f'Command `{command}` on {computer_ip} executed in {time.time() - start_time} seconds. It is completed!'
        
        # output_str = "".join([o["content"] for o in output["output"]])
        # print(f"Command output: {output_str}")
        if killed:
            return {
                'success': False,
                'message': message,
                'computer_ip': computer_ip,
                'session_id': target_session_id,
                'command': command,
                'output': f'Command `{command}` on {computer_ip} executed in {time.time() - start_time} seconds, which surpasses max execution time in `wait_for_completion` mode. Hence, it is killed. If you want to execute commands that will be run for more than 10 seconds, please set `wait_for_completion=False` and use `get_session_output` to get the command content from the session.',
                'async': False,
                'execution_time': time.time() - start_time,
                "timestamp": time.time(),
            }
        return {
            'success': True,
            'message': message,
            'computer_ip': computer_ip,
            'session_id': target_session_id,
            'command': command,
            'output': output["output"],
            'async': False,
            'execution_time': time.time() - start_time,
            "timestamp": time.time()
        }
    
    def run_command_for_env(
        self,
        command: str,
        computer_ip: str = 'localhost',
        session_id: str = None,
        http_port: int = None,
        wait_for_completion: bool = False,
        use_proxy: bool = True
    ) -> Dict[str, Any]:
        """A special run_command for environment setup
        """
        if not command or not command.strip():
            raise ValueError('Command cannot be empty')
        
        # Ensure the target machine has an available session
        if computer_ip not in self.sessions or not self.sessions[computer_ip]:
            # Auto-create session
            try:
                if session_id:
                    create_result = self.create_session(
                        computer_ip=computer_ip, 
                        http_port=http_port,
                        use_proxy=use_proxy,
                        session_id=session_id
                    )
                else:
                    create_result = self.create_session(
                        computer_ip=computer_ip, 
                        http_port=http_port,
                        use_proxy=use_proxy
                    )
                print(f"Auto-created session {create_result['session_id']} on {computer_ip}")
            except Exception as e:
                return {
                    'success': False,
                    'message': f'Failed to create session on {computer_ip}: {str(e)}',
                    'computer_ip': computer_ip,
                    'session_id': None,
                    'command': command,
                    'async': None,
                    'execution_time': 0,
                    'timestamp': time.time(),
                    'output': None,
                }
        
        # Determine which session to use
        target_session_id = session_id or self.default_sessions.get(computer_ip)
        if not target_session_id or target_session_id not in self.sessions[computer_ip]:
            return {
                'success': False,
                'message': f'No valid session found on {computer_ip}',
                'computer_ip': computer_ip,
                'session_id': target_session_id,
                'command': command,
                'async': None,
                'execution_time': 0,
                'timestamp': time.time(),
                'output': None,
            }
        
        session = self.sessions[computer_ip][target_session_id]
        if not session.is_alive():
            # Try to recreate session
            try:
                self.close_session(computer_ip, target_session_id)
                create_result = self.create_session(
                    computer_ip=computer_ip, 
                    http_port=http_port,
                    use_proxy=use_proxy
                )
                if not create_result.get('success'):
                    raise Exception('Failed to recreate session.')
                
                new_session_id = create_result['session_id']
                session = self.sessions[computer_ip][new_session_id]
                target_session_id = new_session_id
                print(f"Recreated session {new_session_id} on {computer_ip}")
            except Exception as e:
                return {
                    'success': False,
                    'message': f'Session {target_session_id} on {computer_ip} is not alive and failed to recreate: {str(e)}',
                    'computer_ip': computer_ip,
                    'session_id': target_session_id,
                    'command': command,
                    'async': None,
                    'execution_time': 0,
                    'timestamp': time.time(),
                    'output': None,
                }
        
        # Ensure terminal has no other processes running
        # print(f"check_shell_children: {session.check_shell_children()}")
        if not session.check_shell_children()['completed']:
            return {
                'success': False,
                'message': 'Current session is busy, there are three options for you to choose:\n(1) Wait last command to finish\n(2) Kill this process, but be careful\n(3) Create a new session to run this command',
                'computer_ip': computer_ip,
                'session_id': target_session_id,
                'command': command,
                'async': None,
                'execution_time': 0,
                'timestamp': time.time(),
                'output': None,
            }


        # Record timestamp before sending command
        start_time = time.time()
        session_timestamp = session.get_session_timestamp()
        # Send command
        success = session.send_command(command)
        if not success:
            return {
                'success': False,
                'message': 'Failed to send command to session',
                'computer_ip': computer_ip,
                'session_id': target_session_id,
                'command': command,
                'async': None,
                'execution_time': 0,
                'timestamp': time.time(),
                'output': None,
            }
        
        if not wait_for_completion:
            # Async execution, return immediately
            return {
                'success': True,
                'message': f'Command sent to session {target_session_id} on {computer_ip}',
                'computer_ip': computer_ip,
                'session_id': target_session_id,
                'command': command,
                'async': True,
                'execution_time': time.time() - start_time,
                "timestamp": time.time(),
                'output': None,
            }
        
        # Wait for command completion
        time.sleep(1)
        killed = False
        while not session.check_command_completion()['completed']:
            if time.time() - start_time >= 10.5:
                self.kill_session_processes(computer_ip, target_session_id, force=True)
                killed = True
                break
            time.sleep(0.5)

        time.sleep(1) # Wait 1 second to give session time to process
        
        
        output = self.get_session_output(computer_ip, target_session_id, since_timestamp=session_timestamp)
        
        if killed:
            message = f'Command {command} on {computer_ip} executed in {time.time() - start_time} seconds, surpass max execution time in wait_for_completion mode, killed. If you want to run commands longer than 10 seconds, please set `wait_for_completion` to False and use `get_session_output` or `get_recent_session_output` to get the connmand content from the session.'
        else:
            message = f'Command {command} on {computer_ip} executed in {time.time() - start_time} seconds, completed'
        
        # output_str = "".join([o["content"] for o in output["output"]])
        # print(f"Command output: {output_str}")
        if killed:
            return {
                'success': False,
                'message': message,
                'computer_ip': computer_ip,
                'session_id': target_session_id,
                'command': command,
                'output': output["output"],
                'async': False,
                'execution_time': time.time() - start_time,
                "timestamp": time.time(),
            }
        return {
            'success': True,
            'message': message,
            'computer_ip': computer_ip,
            'session_id': target_session_id,
            'command': command,
            'output': output["output"],
            'async': False,
            'execution_time': time.time() - start_time,
            "timestamp": time.time()
        }

    def input_in_session(self, computer_ip: str = 'localhost', session_id: str = None,
                        input_text: str = None) -> Dict[str, Any]:
        """Send stdin text to a specific terminal session.
        
        This is valid only when the session's foreground command is awaiting input. If the session is not awaiting input, 
        the request is rejected.
        
        Args:
            computer_ip: str: The IP address of the computer. Default is 'localhost'.
            session_id: str: Unique identifier of the target session. Must refer to an existing, active session. 
                             If the session has no foreground command awaiting stdin, the call will be rejected.
                             Default is None.
            input_text: str: Text to write to the process's stdin. Default is ''.
            
        Returns:
            Dict[str, Any]: Dictionary containing input operation status.
        """
        if not input_text:
            return {
                'success': False,
                'message': 'Input text cannot be empty',
                'computer_ip': computer_ip,
                'session_id': session_id,
                'input_text': input_text,
                'input_check': None,
            }
        
        # Determine which session to use
        target_session_id = session_id or self.default_sessions.get(computer_ip)
        if computer_ip not in self.sessions or not target_session_id or target_session_id not in self.sessions[computer_ip]:
            return {
                'success': False,
                'message': f'Session {target_session_id} does not exist on {computer_ip}',
                'computer_ip': computer_ip,
                'session_id': target_session_id,
                'input_text': input_text,
                'input_check': None,
            }
        
        session = self.sessions[computer_ip][target_session_id]
        if not session.is_alive():
            return {
                'success': False,
                'message': f'Session {target_session_id} on {computer_ip} is not alive',
                'computer_ip': computer_ip,
                'session_id': target_session_id,
                'input_text': input_text,
                'input_check': None,
            }
        
        # Send input (send_input method internally checks if input is allowed)
        result = session.send_input(input_text)
        
        # Add additional information to result
        result['computer_ip'] = computer_ip
        result['session_id'] = target_session_id
        result['input_text'] = input_text
        if 'message' not in result:
            result['message'] = ""
        if 'input_check' not in result:
            result['input_check'] = None

        return result 
    
    def get_session_output(
        self,
        computer_ip: str = 'localhost',
        session_id: str = None,
        start_lines: int = 50,
        end_lines: int = None,
        since_timestamp: float = None
    ) -> Dict[str, Any]:
        """Retrieve the output buffer of the terminal session identified by `session_id`.
        
        If `since_timestamp` is provided, incremental output since that time is returned; otherwise, output is sliced 
        by line window (`start_lines` required, `end_lines` optional).
        
        Args:
            computer_ip: str: The IP address of the computer. Default is 'localhost'.
            session_id: str: Unique identifier of the target session. The session must exist and be active. Default is None.
            start_lines: int: Start offset counted from the **end of output** (>=2). Effective only when `since_timestamp` 
                              is not set. Usage:
                              - `start_lines=N` only: returns the **last N lines**.
                              - With `end_lines`: returns the slice between `start_lines` and `end_lines`.
                              Default is 50.
            end_lines: int: End offset counted from the **end of output** (>=1). If not specified, this tool will return 
                            content from the `start_lines` to the end of the output. If specified, the slice is 
                            **[start_lines, end_lines)**: inclusive of `start_lines`, exclusive of `end_lines`.
                            Default is None.
            since_timestamp: float: Optional. Fetch output **since** this Unix epoch timestamp (seconds, float). 
                                   When set, it **overrides** `start_lines` and `end_lines`. Default is None.
            
        Returns:
            Dict[str, Any]: Dictionary containing session output and status information.
        """
        if end_lines == "None":
            end_lines = None
        if end_lines == 0:
            end_lines = None
        if since_timestamp == "None":
            since_timestamp = None
        target_session_id = session_id or self.default_sessions.get(computer_ip)
        if computer_ip not in self.sessions or not target_session_id or target_session_id not in self.sessions[computer_ip]:
            return {
                'success': False,
                'message': f"Session {target_session_id} does not exist on {computer_ip}",
                'computer_ip': computer_ip,
                'session_id': target_session_id,
                'output': None,
                'is_alive': False,
                'last_activity': None,
            }
        
        session = self.sessions[computer_ip][target_session_id]
        is_alive = session.is_alive()
        if not isinstance(start_lines,int) or (end_lines is not None and not isinstance(end_lines,int)) or start_lines < 1 or (isinstance(end_lines,int) and 1 > end_lines):
            return {
                'success': False,
                'message': f"start_lines should bigger than 1, end_lines should not be specified or end_lines should be in [1, start_lines - 1]",
                'computer_ip': computer_ip,
                'session_id': target_session_id,
                'output': "",
                'is_alive': is_alive,
                'last_activity': session.last_activity,
            }
        if isinstance(end_lines,int):
            if start_lines < end_lines:
                cache = start_lines
                start_lines = end_lines
                end_lines = cache
            elif start_lines == end_lines:
                end_lines = None
        
        output = session.get_output(start_lines=start_lines, end_lines=end_lines, since_timestamp=since_timestamp)
        
        
        if isinstance(end_lines,int):
            line_message = f"from line {len(output) + end_lines} (counting from the end) to line {end_lines} (counting from the end)."
        else: 
            line_message = f"from line {len(output)} (counting from the end) to the end."
       
        result = "".join([o["content"] for o in output])
        if len(result) > self.MAX_OUTPUT_LENGTH:
            result_length = len(result)
            result = result[-self.MAX_OUTPUT_LENGTH:] + f"\nThe context is too long (about {result_length} characters), only present the last {self.MAX_OUTPUT_LENGTH} characters.\n"
        return {
            'success': True,
            'message': f"Successfully retrieved {len(output)} lines of output " + line_message,
            'computer_ip': computer_ip,
            'session_id': target_session_id,
            'output': result,
            'is_alive': is_alive,
            'last_activity': session.last_activity
        }
    
    def get_session_recent_output(self, computer_ip: str = 'localhost', session_id: str = None, seconds: int = 10) -> Dict[str, Any]:
        """Get the recent output from a specific terminal session output buffer.
        
        Args:
            computer_ip: str: The IP address of the computer. Default is 'localhost'.
            session_id: str: The ID of the session to get the recent output. Default is None.
            seconds: int: The number of seconds to get the recent output from. Default is 10.
            
        Returns:
            Dict[str, Any]: Dictionary containing recent session output and status information.
        """
        target_session_id = session_id or self.default_sessions.get(computer_ip)
        if computer_ip not in self.sessions or not target_session_id or target_session_id not in self.sessions[computer_ip]:
            return {
                'success': False,
                'message': f"Session {target_session_id} does not exist on {computer_ip}",
                'computer_ip': computer_ip,
                'session_id': target_session_id,
                'output': None,
                'is_alive': False,
                'last_activity': None,
            }
        
        session = self.sessions[computer_ip][target_session_id]
        is_alive = session.is_alive()
        output = session.get_recent_output(seconds)
        if len(output) > self.MAX_OUTPUT_LENGTH:
            output_length = len(output)
            output = output[-self.MAX_OUTPUT_LENGTH:] + f"\nThe context is too long (about {output_length} characters), only present the last {self.MAX_OUTPUT_LENGTH} characters.\n"
        return {
            'success': True,
            'message': 'Successfully retrieved recent output.',
            'computer_ip': computer_ip,
            'session_id': target_session_id,
            'output': output,
            'is_alive': is_alive,
            'last_activity': session.last_activity
        }
    
    def session_status(self, computer_ip: str = 'localhost', session_id: str = None) -> Dict[str, Any]:
        """Get the status of a specific terminal session.
        
        Args:
            computer_ip: str: The IP address of the computer. Default is 'localhost'.
            session_id: str: Unique identifier of the target session. If absent, the status of the default session is returned. Default is None.
            
        Returns:
            Dict[str, Any]: Dictionary containing session status information.
        """
        target_session_id = session_id or self.default_sessions.get(computer_ip)
        if computer_ip not in self.sessions or not target_session_id or target_session_id not in self.sessions[computer_ip]:
            return {
                'success': False,
                'message': f"Session {target_session_id} does not exist on {computer_ip}",
                'computer_ip': computer_ip,
                'session_id': target_session_id,
                'status': None,
            }
        
        session = self.sessions[computer_ip][target_session_id]
        status = session.check_shell_children()
        return {
            'success': True,
            'message': 'Successfully retrieved session status.',
            'computer_ip': computer_ip,
            'session_id': target_session_id,
            'status': status
        }
    
    def session_idle(self, computer_ip: str = 'localhost', session_id: str = None) -> Dict[str, Any]:
        """Check if a specific terminal session is idle.
        
        Args:
            computer_ip: str: The IP address of the computer. Default is 'localhost'.
            session_id: str: The ID of the session to check whether it is running some command or whether it is idle.
                             Default is None.
        
        Returns:
            Dict[str, Any]: Dictionary containing session idle status information.
        """
        status_result = self.session_status(computer_ip, session_id)
        if not status_result.get('success', False):
            return {
                'success': False,
                'message': f'{status_result.get("message", "Could not get session status.")}',
                'is_idle': False,
            }
        
        is_idle = status_result.get('status', {}).get('completed', False)
        reason = status_result.get('status', {}).get('reason', '')
        reason = reason.split("\n")
        if len(reason) > 30:
            reason = "\n".join(reason[-30:]) + "\n" + f"[Too many processes: {len(reason)}, only showing last 30.]"
        else:
            reason = "\n".join(reason) 
        return {
            'success': True,
            'message': f"Session is {'idle' if is_idle else 'busy'}, because {reason}",
            'is_idle': is_idle
        }

    def clear_session_buffer(self, computer_ip: str = 'localhost', session_id: str = None) -> Dict[str, Any]:
        """Clear the output buffer of a specific terminal session.
        
        The output buffer is a queue of output lines, it will automatically clean if the total lines exceed 10000 lines,
        regardless of using this action or not.
        
        Args:
            computer_ip: str: The IP address of the computer. Default is 'localhost'.
            session_id: str: The ID of the session to clear the output buffer. Default is None.
            
        Returns:
            Dict[str, Any]: Dictionary containing operation status information.
        """
        target_session_id = session_id or self.default_sessions.get(computer_ip)
        if computer_ip in self.sessions and target_session_id and target_session_id in self.sessions[computer_ip]:
            session = self.sessions[computer_ip][target_session_id]
            clear_result = session.clear_buffer()
            return {
                'success': clear_result,
                'message': 'Session buffer cleared successfully' if clear_result else 'Failed to clear session buffer',
                'computer_ip': computer_ip,
                'session_id': target_session_id,
            }
        else:
            print(f"Session {target_session_id} not found on {computer_ip}")
            return {
                'success': False,
                'message': f'Session {target_session_id} not found on {computer_ip}',
                'computer_ip': computer_ip,
                'session_id': target_session_id,
            }
    
    def close_session(self, computer_ip: str, session_id: str) -> Dict[str, Any]:
        """Close a specific terminal session and kill all sub-processes in the session.
        
        Args:
            computer_ip: str: The IP address of the computer.
            session_id: str: The ID of the session to close.
            
        Returns:
            Dict[str, Any]: Dictionary containing operation status information.
        """
        if computer_ip not in self.sessions or session_id not in self.sessions[computer_ip]:
            return {
                'success': False,
                'message': f"Session {session_id} does not exist on {computer_ip}",
                'computer_ip': computer_ip,
                'session_id': session_id,
                'num_closed_sessions': 0,
                'closed_machines': [],
            }
        
        try:
            with self.lock:
                session = self.sessions[computer_ip][session_id]
                session.terminate()
                del self.sessions[computer_ip][session_id]
                
                # If closing the default session for this machine, choose a new default session
                if self.default_sessions.get(computer_ip) == session_id:
                    if self.sessions[computer_ip]:
                        self.default_sessions[computer_ip] = next(iter(self.sessions[computer_ip].keys()))
                    else:
                        del self.default_sessions[computer_ip]
                
                # If no sessions left on this machine, clean up
                if not self.sessions[computer_ip]:
                    del self.sessions[computer_ip]
            
            return {
                'success': True,
                'message': f'Session {session_id} closed successfully on {computer_ip}',
                'computer_ip': computer_ip,
                'session_id': session_id,
                'num_closed_sessions': 1,
                'closed_machines': [computer_ip],
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to close session {session_id} on {computer_ip}: {str(e)}',
                'computer_ip': computer_ip,
                'session_id': session_id,
                'num_closed_sessions': 0,
                'closed_machines': [],
            }
    
    def close_all_sessions(self, computer_ip: str = None) -> Dict[str, Any]:
        """Close all sessions on a specific machine or all machines.
        
        If you want to close all sessions on a specific machine, you should set the `computer_ip`.
        
        Args:
            computer_ip: str: The IP address of the computer. If None, closes sessions on all machines. Default is None.
            
        Returns:
            Dict[str, Any]: Dictionary containing operation status information.
        """
        try:
            closed_sessions_count = 0
            closed_machines = []
            
            with self.lock:
                if computer_ip:
                    # Only close sessions for specified machine
                    if computer_ip in self.sessions:
                        session_count = len(self.sessions[computer_ip])
                        for session in self.sessions[computer_ip].values():
                            session.terminate()
                        del self.sessions[computer_ip]
                        if computer_ip in self.default_sessions:
                            del self.default_sessions[computer_ip]
                        closed_sessions_count = session_count
                        closed_machines = [computer_ip]
                    else:
                        return {
                            'success': True,
                            'message': f'No sessions found on {computer_ip}',
                            'computer_ip': computer_ip,
                            'num_closed_sessions': 0,
                            'closed_machines': [],
                        }
                else:
                    # Close sessions for all machines
                    for machine_ip, machine_sessions in self.sessions.items():
                        for session in machine_sessions.values():
                            session.terminate()
                        closed_sessions_count += len(machine_sessions)
                        closed_machines.append(machine_ip)
                    self.sessions.clear()
                    self.default_sessions.clear()
            
            return {
                'success': True,
                'message': f'Successfully closed {closed_sessions_count} sessions on {len(closed_machines)} machines',
                'computer_ip': computer_ip,
                'num_closed_sessions': closed_sessions_count,
                'closed_machines': closed_machines,
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to close sessions: {str(e)}',
                'computer_ip': computer_ip,
                'num_closed_sessions': 0,
                'closed_machines': [],
            }
    
    def kill_session_processes(self, computer_ip: str = 'localhost', session_id: str = None,
                              force: bool = False) -> Dict[str, Any]:
        """Kill all processes on a specific session.
        
        Args:
            computer_ip: str: The IP address of the computer. Default is 'localhost'.
            session_id: str: The ID of the session to kill all processes. Default is None.
            force: bool: Whether to force to kill all processes. Default is False.
            
        Returns:
            Dict[str, Any]: Dictionary containing operation status information.
        """
        target_session_id = session_id or self.default_sessions.get(computer_ip)
        if computer_ip not in self.sessions or not target_session_id or target_session_id not in self.sessions[computer_ip]:
            return {
                'success': False,
                'message': f"Session {target_session_id} does not exist on {computer_ip}",
                'computer_ip': computer_ip,
                'session_id': target_session_id,
                'signal_used': 'KILL' if force else 'TERM',
                'killed_processes': [],
                'failed_kills': [],
            }
        
        session = self.sessions[computer_ip][target_session_id]
        
        # Call session's kill_processes method
        result = session.kill_processes(force=force)

        # Add computer_ip and session_id information to result
        result['computer_ip'] = computer_ip
        result['session_id'] = target_session_id
        result['signal_used'] = 'KILL' if force else 'TERM'
        result['killed_processes'] = result.pop('killed_processes', [])
        result['failed_kills'] = result.pop('failed_kills', [])
        
        return result
    
    def __del__(self):
        """Destructor, ensures all sessions are cleaned up"""
        self.close_all_sessions()
