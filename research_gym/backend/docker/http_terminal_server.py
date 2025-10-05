#!/usr/bin/env python3
"""
HTTP Terminal Server
Run this service on a remote machine to provide HTTP API for executing commands
No authentication required, simple and easy to use
"""

import os
import signal
import threading
import time
import uuid
import queue
import ptyprocess
import json
from flask import Flask, request, jsonify
from typing import Dict, Any, List
from collections import deque


class TerminalSession:
    """Single terminal session class, supports local connections"""
    
    def __init__(self, session_id: str, shell: str = '/bin/bash', buffer_size: int = 10000):
        self.session_id = session_id
        self.shell = shell
        self.process = None
        self.output_buffer = deque(maxlen=buffer_size)  # Output buffer, stores up to 10000 lines
        self.is_running = False
        self.current_dir = os.getcwd()
        self.lock = threading.Lock()
        self.output_thread = None
        self.command_queue = queue.Queue()
        self.last_activity = time.time()
        self.created_at = time.time()  # Add creation time
        # Add command completion detection related attributes
        self.command_completion_marker = None
        self.last_command_timestamp = None
        self.command_in_progress = False
        
    def start(self):
        """Start local terminal session"""
        try:
            # Create pseudo-terminal process
            self.process = ptyprocess.PtyProcess.spawn([self.shell])
            self.process.setwinsize(100, 300)
            self.is_running = True
            
            # Start output reading thread
            self.output_thread = threading.Thread(target=self._read_output, daemon=True)
            self.output_thread.start()
            
            # Wait for shell to start up
            time.sleep(0.1)
            
            return True
        except Exception as e:
            self.is_running = False
            raise Exception(f"Failed to start local terminal session: {str(e)}")
    
    def _read_output(self):
        """Read terminal output in background thread"""
        while self.is_running and self.process and self.process.isalive():
            try:
                # PtyProcess.read() does not support timeout parameter, use non-blocking read
                output = self.process.read()
                if output:
                    with self.lock:
                        # Split output by lines and add to buffer
                        lines = output.decode('utf-8', errors='replace').splitlines(True)
                        for line in lines:
                            self.output_buffer.append({
                                'timestamp': time.time(),
                                'content': line
                            })
                        self.last_activity = time.time()
                
                # Reduce delay, improve responsiveness
                time.sleep(0.05)
                
            except (EOFError, OSError):
                # Process ended or connection disconnected
                break
            except Exception as e:
                print(f"Error reading output from session {self.session_id}: {e}")
                break

    def get_session_timestamp(self) -> float:
        """Get current timestamp"""
        return time.time()
     
    def send_command(self, command: str) -> bool:
        """Send command to terminal"""
        if not self.is_running or not self.process or not self.process.isalive():
            return False
        
        try:
            # Mark command execution start
            with self.lock:
                self.command_in_progress = True
                self.last_command_timestamp = time.time()
                # Generate unique completion marker
                self.command_completion_marker = f"CMD_COMPLETE_{uuid.uuid4().hex[:8]}"
            
            # Ensure terminal is ready
            time.sleep(0.05)
            
            # For very long commands, use chunked sending
            if len(command) > 500:  # Commands over 500 characters use chunked sending
                return self._send_long_command(command)
            else:
                # Send short commands directly
                self.process.write((command + '\n').encode('utf-8'))
            self.process.flush()
            self.last_activity = time.time()
            return True
        except Exception as e:
            print(f"Error sending command to session {self.session_id}: {e}")
            return False
    
    def _send_long_command(self, command: str) -> bool:
        """Send long commands in chunks to avoid buffer overflow"""
        try:
            command_bytes = (command + '\n').encode('utf-8')
            chunk_size = 512  # 512 byte chunk size, more conservative
            
            for i in range(0, len(command_bytes), chunk_size):
                chunk = command_bytes[i:i + chunk_size]
                self.process.write(chunk)
                # Brief delay to ensure data is processed
                time.sleep(0.02)
            
            return True
        except Exception as e:
            print(f"Error sending long command to session {self.session_id}: {e}")
            return False
    
    def get_output(self, start_lines: int = 50, end_lines: int = None, since_timestamp: float = None) -> List[Dict]:
        """Get terminal output"""
        with self.lock:
            if since_timestamp:
                # Get output after specified timestamp
                result = [item for item in self.output_buffer 
                         if item['timestamp'] > since_timestamp]
            else:
                # Get the most recent specified number of lines
                result = list(self.output_buffer)[-start_lines:] if start_lines > 0 else list(self.output_buffer)
                if end_lines and len(result) > end_lines:
                    result = result[:-end_lines]
            
            return result
    
    def get_recent_output(self, seconds: int = 10) -> str:
        """Get recent seconds of output as string"""
        cutoff_time = time.time() - seconds
        recent_items = self.get_output(since_timestamp=cutoff_time)
        return ''.join([item['content'] for item in recent_items])
    
    def clear_buffer(self):
        """Clear output buffer"""
        with self.lock:
            self.output_buffer.clear()

    
    def _get_recent_output_content(self, seconds: int) -> str:
        """Get recent seconds of output content"""
        cutoff_time = time.time() - seconds
        with self.lock:
            recent_items = [item for item in self.output_buffer 
                           if item['timestamp'] > cutoff_time]
            return ''.join([item['content'] for item in recent_items])
    
    def is_alive(self) -> bool:
        """Check if session is still alive"""
        return self.is_running and self.process and self.process.isalive()
    
    def terminate(self):
        """Terminate session"""
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
        """Check if shell has child processes running"""
        if not self.is_alive():
            return {
                'completed': True,
                'reason': 'the session is dead.',
                'children_count': 0
            }
        
        # If no command is executing, return completed status directly
        if not self.command_in_progress:
            return {
                'completed': True,
                'reason': 'there is no command in progress.',
                'children_count': 0
            }
        
        try:
            # Get the PID of shell process
            shell_pid = self.process.pid
            
            # First check if shell process itself still exists
            import subprocess
            shell_check = subprocess.run(
                ['ps', '-p', str(shell_pid)],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if shell_check.returncode != 0:
                # Shell process doesn't exist, meaning it has exited and commands are definitely completed
                with self.lock:
                    self.command_in_progress = False
                    self.command_completion_marker = None
                
                return {
                    'completed': True,
                    'reason': 'the shell process has exited.',
                    'children_count': 0,
                    'children_info': []
                }
            
            # Shell process exists, continue checking child processes
            # Use recursive method to get all child processes (including grandchildren)
            def get_all_children(parent_pid):
                """Recursively get all child processes"""
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
                            
                            # Recursively check this process's child processes
                            grandchildren = get_all_children(pid)
                            children.extend(grandchildren)
                            
                except Exception:
                    pass
                
                return children
            
            # Get all child processes
            all_children = get_all_children(shell_pid)
            
            # Stricter filtering: only focus on actual command execution processes
            actual_children = []
            children_info_output = ""
            for pid, ppid, stat, cmd in all_children:
                
                # Filter out irrelevant processes
                # Use more precise matching conditions
                should_exclude = False
                
                # Filter out ps command itself
                if cmd.startswith('ps ') and '--ppid' in cmd:
                    should_exclude = True
                
                # Filter out pure bash processes (empty bash without running scripts)
                elif cmd.strip() in ['/bin/bash', 'bash', '-bash']:
                    should_exclude = True
                
                # Filter out kernel threads (process names starting and ending with brackets)
                elif cmd.startswith('[') and cmd.endswith(']') and ' ' not in cmd.strip('[]'):
                    should_exclude = True
                
                if should_exclude:
                    continue
                
                # Check process state, filter out zombie and stopped processes
                if 'Z' in stat or 'T' in stat:  # Z=zombie, T=stopped
                    continue
                
                actual_children.append(f"{pid} {stat} {cmd}")
                children_info_output += f"- Process {pid} is in {stat} status, running command `{cmd}`.\n"
            
            children_count = len(actual_children)
            
            # If there are no actual child processes, the command may be completed
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
            # Throw exception for debugging when error occurs
            # raise Exception(f"error: {str(e)}")
            return {
                'completed': False,
                'reason': f'error: {str(e)}',
                'children_count': 0,
                'children_info': []
            }
    
    def check_command_completion(self) -> Dict[str, Any]:
        """Check if current command is completed - prioritize child process detection"""
        if not self.command_in_progress:
            return {
                'completed': True,
                'reason': 'there is no command in progress.'
            }
        
        if not self.is_alive():
            return {
                'completed': True,
                'reason': 'the session is dead.'
            }
        
        # Prioritize child process detection method
        return self.check_shell_children()

    def kill_processes(self, force: bool = False) -> Dict[str, Any]:
        """
        Kill all child processes running in the session
        
        Args:
            force: Whether to use SIGKILL to force kill (default to SIGTERM first)
            
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
                    
                    # Use kill command to kill process
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
            
            # If using TERM signal, wait and check if there are still processes that need force killing
            if not force and killed_processes:
                time.sleep(1)  # Wait for processes to exit gracefully
                
                # Check again if there are still child processes
                remaining_check = self.check_shell_children()
                if not remaining_check.get('completed', True):
                    remaining_children = remaining_check.get('children_info', [])
                    if remaining_children:
                        # Some processes haven't exited, use SIGKILL to force kill
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
            
            # Mark command execution completed
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

    
    def check_waiting_for_input(self, no_output_seconds: int = 20) -> Dict[str, Any]:
        """Check if session is waiting for user input"""
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
                # Get timestamp of most recent output
                recent_output_time = self.output_buffer[-1]['timestamp']
        
        current_time = time.time()
        seconds_since_output = current_time - recent_output_time
        
        # If no output for more than specified time, consider it may be waiting for input
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
            # Send input (automatically add newline character)
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
                'error': f"Failed to send input: {str(e)}",
                'message': f"Failed to send input: {str(e)}",
            }

class HTTPTerminalServer:
    """HTTP Terminal Server"""
    
    def __init__(self, port: int = 8123):
        self.port = port
        self.sessions: Dict[str, TerminalSession] = {}
        self.lock = threading.Lock()
        self.app = Flask(__name__)
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup routes"""
        
        @self.app.route('/api/sessions', methods=['POST'])
        def create_session():
            """Create new session"""
            try:
                data = request.get_json() or {}
                session_id = data.get('session_id') or str(uuid.uuid4())[:8]
                shell = data.get('shell', '/bin/bash')
                
                if session_id in self.sessions:
                    return jsonify({
                        'success': False,
                        'error': f'Session {session_id} already exists'
                    }), 400
                
                with self.lock:
                    session = TerminalSession(session_id, shell)
                    session.start()
                    self.sessions[session_id] = session
                
                return jsonify({
                    'success': True,
                    'session_id': session_id,
                    'message': 'Session created successfully'
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/sessions/<session_id>', methods=['DELETE'])
        def delete_session(session_id):
            """Delete session"""
            try:
                if session_id not in self.sessions:
                    return jsonify({
                        'success': False,
                        'error': f'Session {session_id} not found'
                    }), 404
                
                with self.lock:
                    session = self.sessions[session_id]
                    session.terminate()
                    del self.sessions[session_id]
                
                return jsonify({
                    'success': True,
                    'message': f'Session {session_id} deleted'
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/sessions/<session_id>/command', methods=['POST'])
        def send_command(session_id):
            """Send command to session"""
            try:
                if session_id not in self.sessions:
                    return jsonify({
                        'success': False,
                        'error': f'Session {session_id} not found'
                    }), 404
                
                data = request.get_json() or {}
                command = data.get('command')
                
                if not command:
                    return jsonify({
                        'success': False,
                        'error': 'Command is required'
                    }), 400
                
                session = self.sessions[session_id]
                if not session.is_alive():
                    return jsonify({
                        'success': False,
                        'error': f'Session {session_id} is not alive'
                    }), 400
                
                if not session.check_shell_children()['completed']:
                    return jsonify({
                        'success': False,
                        'error': f'Terminal is not empty, please (1) wait last command to finish or (2) kill this process or (3) establish a new session',
                    }), 400
                
                success = session.send_command(command)
                
                return jsonify({
                    'success': success,
                    'message': 'Command sent' if success else 'Failed to send command'
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/sessions/<session_id>/output', methods=['GET'])
        def get_output(session_id):
            """Get session output"""
            try:
                if session_id not in self.sessions:
                    return jsonify({
                        'success': False,
                        'error': f'Session {session_id} not found'
                    }), 404
                
                start_lines = request.args.get('start_lines', type=int, default=50)
                end_lines = request.args.get('end_lines', type=int)
                since_timestamp = request.args.get('since_timestamp', type=float)
                session = self.sessions[session_id]
                output = session.get_output(start_lines=start_lines, end_lines=end_lines, since_timestamp=since_timestamp)
                
                return jsonify({
                    'success': True,
                    'output': output,
                    'session_id': session_id
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/sessions/<session_id>/output', methods=['DELETE'])
        def clear_output_buffer(session_id):
            """Clear session's output buffer"""
            try:
                if session_id not in self.sessions:
                    return jsonify({
                        'success': False,
                        'error': f'Session {session_id} not found'
                    }), 404
                
                session = self.sessions[session_id]
                session.clear_buffer()
                
                return jsonify({
                    'success': True,
                    'session_id': session_id,
                    'message': 'Output buffer cleared successfully'
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/sessions/<session_id>/status', methods=['GET'])
        def get_status(session_id):
            """Get session status"""
            try:
                if session_id not in self.sessions:
                    return jsonify({
                        'success': False,
                        'error': f'Session {session_id} not found'
                    }), 404
                
                session = self.sessions[session_id]
                
                return jsonify({
                    'success': True,
                    'session_id': session_id,
                    'is_alive': session.is_alive(),
                    'last_activity': session.last_activity,
                    'created_at': session.created_at,
                    'current_dir': session.current_dir
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/sessions/<session_id>/shell_children', methods=['GET'])
        def get_shell_children(session_id):
            """Get shell child process information in session"""
            try:
                if session_id not in self.sessions:
                    return jsonify({
                        'success': False,
                        'error': f'Session {session_id} not found'
                    }), 404
                
                session = self.sessions[session_id]
                children_status = session.check_shell_children()
                
                return jsonify({
                    'success': True,
                    'session_id': session_id,
                    'children_status': children_status
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/sessions/<session_id>/command_completion', methods=['GET'])
        def check_command_completion(session_id):
            """Check if command in session is completed"""
            try:
                if session_id not in self.sessions:
                    return jsonify({
                        'success': False,
                        'error': f'Session {session_id} not found'
                    }), 404
                
                session = self.sessions[session_id]
                completion_status = session.check_command_completion()
                
                return jsonify({
                    'success': True,
                    'session_id': session_id,
                    'completion_status': completion_status
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/sessions/<session_id>/kill_processes', methods=['POST'])
        def kill_processes(session_id):
            """Kill all child processes running in session"""
            try:
                if session_id not in self.sessions:
                    return jsonify({
                        'success': False,
                        'error': f'Session {session_id} not found'
                    }), 404
                
                data = request.get_json() or {}
                force = data.get('force', False)
                
                session = self.sessions[session_id]
                result = session.kill_processes(force=force)
                
                # Add session_id to result
                result['session_id'] = session_id
                
                return jsonify(result)
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'session_id': session_id
                }), 500
        
        @self.app.route('/api/sessions/<session_id>/check_input', methods=['GET'])
        def check_waiting_for_input(session_id):
            """Check if session is waiting for user input"""
            try:
                if session_id not in self.sessions:
                    return jsonify({
                        'success': False,
                        'error': f'Session {session_id} not found'
                    }), 404
                
                no_output_seconds = request.args.get('no_output_seconds', type=int, default=20)
                
                session = self.sessions[session_id]
                input_status = session.check_waiting_for_input(no_output_seconds=no_output_seconds)
                
                return jsonify({
                    'success': True,
                    'session_id': session_id,
                    'input_status': input_status
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'session_id': session_id
                }), 500
        
        @self.app.route('/api/sessions/<session_id>/input', methods=['POST'])
        def send_input(session_id):
            """Send input to session"""
            try:
                if session_id not in self.sessions:
                    return jsonify({
                        'success': False,
                        'message': f'Session {session_id} not found'
                    }), 404
                
                data = request.get_json() or {}
                input_text = data.get('input')
                
                if not input_text:
                    return jsonify({
                        'success': False,
                        'message': 'Input text is required'
                    }), 400
                
                session = self.sessions[session_id]
                result = session.send_input(input_text)
                
                # Add session_id to result
                result['session_id'] = session_id
                
                return jsonify(result)
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': str(e),
                    'session_id': session_id
                }), 500
        
        @self.app.route('/api/sessions', methods=['GET'])
        def list_sessions():
            """List all sessions"""
            try:
                result = {}
                with self.lock:
                    for sid, session in self.sessions.items():
                        result[sid] = {
                            'session_id': sid,
                            'is_alive': session.is_alive(),
                            'last_activity': session.last_activity,
                            'created_at': session.created_at,
                            'current_dir': session.current_dir
                        }
                
                return jsonify({
                    'success': True,
                    'sessions': result
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check"""
            return jsonify({
                'status': 'healthy',
                'timestamp': time.time(),
                'active_sessions': len(self.sessions)
            })
        
        @self.app.route('/timestamp', methods=['GET'])
        def get_session_timestamp():
            """Get current timestamp"""
            return jsonify({
                'timestamp': time.time()
            })
            
    
    def start_server(self, host: str = '0.0.0.0', debug: bool = False):
        """Start HTTP server"""
        print(f"Starting HTTP Terminal Server on {host}:{self.port}")
        print("No authentication required - server is open to all requests")
        print("Make sure to use this only in trusted environments!")
        
        self.app.run(host=host, port=self.port, debug=debug, threaded=True)
    
    def cleanup(self):
        """Cleanup all sessions"""
        with self.lock:
            for session in self.sessions.values():
                session.terminate()
            self.sessions.clear()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='HTTP Terminal Server (No Authentication)')
    parser.add_argument('--port', '-p', type=int, default=8123, help='Server port (default: 8123)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    server = HTTPTerminalServer(port=args.port)
    
    try:
        server.start_server(host=args.host, debug=args.debug)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.cleanup()
    except Exception as e:
        print(f"Server error: {e}")
        server.cleanup()


if __name__ == '__main__':
    main() 