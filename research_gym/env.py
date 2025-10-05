import subprocess
import time
from datetime import datetime
import os
import logging
from typing import Optional, Any

from tqdm import tqdm

from research_gym.action import BaseAction, RunCommandAction
from research_gym.action.empty import NullAction
from research_gym.action.system import FinishAction
from research_gym.observation.observation import BaseObservation
from research_gym.configs.task_config import TaskConfig, ComputerType
from research_gym.action.action_manager import ActionManager
from research_gym.base_node import Source
from agents.context.nodes.base_node import BaseNode
from research_gym.applications.cmd_operations import CmdOperations


class BaseEnv:
    def __init__(self, task_config: TaskConfig):
        self.task_config = task_config
        
        BaseObservation.actual_workspace = self.task_config.actual_workspace
        # Ensure actual_workspace exists

        os.makedirs(self.task_config.actual_workspace, exist_ok=True)
        
        # Initialize nodes and backup directory
        folder = self.build_workspace_backup_folder()
        BaseNode.directory = folder + "/nodes"
        os.makedirs(BaseNode.directory, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize action manager
        self.action_manager = ActionManager(self.task_config)
        
        # Initialize action manager
        self.action_manager.initialize()
        self.conda_dir_mapping = {
            "task_1": "llamafactory",
            "task_2": "llamafactory",
            "task_3": "llamafactory",
            "task_4": "llamafactory",
            "task_5": "vllm",
            "task_6": "llamafactory",
            "task_7": "vllm",
            "task_8": "llamafactory",
            "task_9": "llamafactory",
            "task_10": "llamafactory_retriever",
            "task_11": "llamafactory",
            "task_12": "llamafactory",
            "task_13": "llamafactory",
            "task_14": "dapo",
            "task_15": "alignment-handbook",
            "task_16": "search_r1_verl_retriever",
            "task_17": "guir1_verl",
            "task_18": "vllm_retriever",
            "task_19": "visualsketchpad",
            "task_20": "visualsketchpad",
        }

    def build_workspace_backup_folder(self) -> str:
        """
        Build and ensure the existence of workspace backup directory for saving tree and node data.
        High cohesion: directory construction logic is centralized in the environment layer;
        Low coupling: external access only through this method to get the path and use it.
        """
        folder = f"{self.task_config.checkpoint_base_path}/workspace_backup/workspace_{self.task_config.task_name}_{self.task_config.start_time.strftime('%Y-%m-%d.%H:%M:%S')}_step{self.task_config.max_steps}_{self.task_config.model_name}"
        os.makedirs(folder, exist_ok=True)
        return folder

    def reset(self) -> BaseObservation:
        if self.task_config.launch_type == 'direct_http':
            self._reset_by_direct_http()
            pass
        elif self.task_config.launch_type == "load_checkpoint":
            self._reset_by_load_checkpoint()
            pass
        else:
            raise ValueError(f"Unsupported launch type: {self.task_config.launch_type}")
        
        return self._retrieve_task_description()
    
    def _retrieve_task_description(self) -> BaseObservation:
        if self.task_config.launch_type == 'direct_http':
            # Read /workspace/task/task_description.md
            with open('/workspace/task/task_description.md', 'r', encoding='utf-8') as file:
                task_description = file.read()

            obs = BaseObservation()
            obs._message = task_description
            obs._source = Source.USER  
            obs.timestamp = datetime.now()
            return obs
        elif self.task_config.launch_type == "load_checkpoint":
            return BaseObservation()
        else:
            raise ValueError(f"Unsupported launch type: {self.task_config.launch_type}")
            
    def _reset_by_direct_http(self) -> BaseObservation:
        """
        Reset environment using direct HTTP connection
        """
        pass
    
    def _reset_by_load_checkpoint(self) -> BaseObservation:
        """
        Restore environment from checkpoint
        """
        pass
    
    def _redirect_localhost_action(self, action: RunCommandAction) -> Optional[BaseObservation]:
        """
        If action points to localhost, redirect it to the main container.
        If successful, modify the action object.
        If configuration is missing, return an error observation.
        """
        container_ip, container_port = None, None
        
        if self.task_config.launch_type == 'direct_http':
            if not self.task_config.computer_pool:
                self.logger.error("⚠️ Security interception: No available HTTP endpoints, refusing to execute local commands")
                return self._create_error_observation("Security interception: No available HTTP endpoints")
            # Note: Here we use the first configured HTTP endpoint, actual scenarios may need more complex selection logic
            http_config = self.task_config.computer_pool[0]
            container_ip = http_config.computer_ip
            container_port = http_config.computer_port

        if container_ip and container_port:
            action.computer_ip = container_ip
            if not action.http_port:
                action.http_port = container_port
            self.logger.info(f"🔒 Security interception: Redirecting command from local to container {container_ip}:{container_port}")
            print(f"🔒 Security interception: Redirecting command from local to container {container_ip}:{container_port}")
        
        return None

    def step(self, action: BaseAction) -> BaseObservation:
        """
        Execute action and return observation result

        Args:
            action: The action to execute

        Returns:
            BaseObservation: Observation of execution result
        """
        from research_gym.action.commands import RunCommandAction
        # Security interception: Force all RunCommandAction to execute in container
        if isinstance(action, RunCommandAction):
            # If Agent specifies localhost or doesn't specify container IP, force to set container IP
            if action.computer_ip == 'localhost' or action.computer_ip == '127.0.0.1':
                error_obs = self._redirect_localhost_action(action)
                if error_obs:
                    return error_obs
        elif isinstance(action, NullAction):
            # Agent has no tool call
            if action.call_id == "00000000":
                # Agent has tool call, but no corresponding tool
                # Create a basic observation result for NullAction
                result = {
                    'success': False,
                    'message': 'No action performed',
                    'timestamp': datetime.now().timestamp(),
                    'source': Source.ENVIRONMENT.value
                }
                obs = BaseObservation.from_result(result, action)
                return obs
            else:
                result = {
                    'success': False,
                    'message': action.error_message,
                    'timestamp': datetime.now().timestamp(),
                }
                obs = BaseObservation.from_result(result, action)
                return obs
        elif isinstance(action, FinishAction):
            result = {
                'success': True,
                'message': "Task is done!",
                'timestamp': datetime.now().timestamp(),
                'source': Source.ENVIRONMENT.value,
            }
            obs = BaseObservation.from_result(result, action)
            return obs
        # return self.action_manager.execute_action(action)
        try:
            # Execute action using ActionManager
            return self.action_manager.execute_action(action)
        except Exception as e:
            self.logger.error(f"Error occurs when executing the action {action.action_type.value}: {str(e)}")
            return self._create_error_observation(f'Error occurs when executing the action {action.action_type.value}: {str(e)}', action)

    def _create_error_observation(self, message: str, action: BaseAction = None) -> BaseObservation:
        """Create error observation result"""
        try:
            obs = BaseObservation.from_result({
                'success': False,
                'message': message,
                'output': {'output': []}
            }, action)
            return obs
        except Exception as e:
            self.logger.error(f"Error occurred when creating error observation result: {str(e)}")
            null_action = NullAction()
            return BaseObservation.from_result({
                'success': False,
                'message': message,
                'output': {'output': []}
            }, null_action)

    def render(self):
        pass

    def close(self):
        pass


class GZEnv(BaseEnv):
    def __init__(self, task_config: TaskConfig):
        super().__init__(task_config)
        self.workspace_dataset_path = task_config.workspace_dataset_path
        self.actual_workspace = task_config.actual_workspace
        
    def _reset_by_direct_http(self) -> BaseObservation:
        """
        Reset environment using direct HTTP connection
        Need to know GPU IP and Port in advance, located in TaskConfig's computer_pool
        computer_pool: [{
            computer_ip: "10.244.203.11",
            computer_port: 8123,
            available: True
        },
        {
            computer_ip: "10.244.38.140",
            computer_port: 8123,
            available: True
        }]
        """
        RESET_DONE = False
        MAX_WAIT_TIME = 60
        if not self.task_config.computer_pool:
            self.logger.error("No available HTTP ports, please check configuration")
            raise ValueError("No available HTTP ports, please check configuration")
        
        available_http_computer_idxs = [idx for idx in range(len(self.task_config.computer_pool)) if self.task_config.computer_pool[idx].available]
        
        # Move LOCALHOST_CPU to the front
        available_http_computer_idxs_cache =  available_http_computer_idxs.copy()
        available_http_computer_idxs = []
        for available_http_computer_idx in available_http_computer_idxs_cache:
            computer = self.task_config.computer_pool[available_http_computer_idx]
            computer_type = computer.computer_type
            if computer_type == ComputerType.LOCALHOST_CPU:
                available_http_computer_idxs.append(available_http_computer_idx)
                
        for available_http_computer_idx in available_http_computer_idxs_cache:
            computer = self.task_config.computer_pool[available_http_computer_idx]
            computer_type = computer.computer_type
            if computer_type == ComputerType.LOCALHOST_CPU:
                pass
            else:
                available_http_computer_idxs.append(available_http_computer_idx)
                
        start_time = time.time()
        while not RESET_DONE and time.time() < start_time + MAX_WAIT_TIME:
            # print(available_http_computer_idxs)
            for idxx, idx in enumerate(available_http_computer_idxs):
                computer = self.task_config.computer_pool[idx]
                computer_ip = computer.computer_ip
                computer_port = computer.computer_port
                computer_type = computer.computer_type
                if idxx == 0:
                    assert computer_type == ComputerType.LOCALHOST_CPU, "The index of LOCALHOST_CPU must be the 0."
                if computer_type == ComputerType.LOCALHOST_CPU:
                    # Clear workspace in parallel
                    rm_commands = [
                        f"rm -rf /workspace",
                        f"rm -rf {self.actual_workspace}/*",
                        f"rm -rf {self.actual_workspace}/.[!.]*",
                        f"rm -rf {self.actual_workspace}/..?*"
                    ]
                    processes = [subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for cmd in rm_commands]
                    
                    exit_codes = []
                    print("Start parallel deletion of task files...")
                    for p in tqdm(processes, desc="Delete task files", unit="tasks"):
                        exit_codes.append(p.wait())
                    if any(code != 0 for code in exit_codes):
                        self.logger.error(f"One or more deletion commands failed. Please check permissions and paths.")
                        raise ValueError(f"One or more deletion commands failed. Please check permissions and paths.")
                    
                    # Use subprocess
                    # Create soft link
                    link_command = f"mkdir -p {self.actual_workspace} && ln -s {self.actual_workspace} /workspace"
                    link_process = subprocess.run(link_command, shell=True, capture_output=True, text=True)
                    if link_process.returncode != 0 and "File exists" not in link_process.stderr:
                        self.logger.error(f"Failed to create soft link /workspace: {link_process.stderr}")
                        raise ValueError(f"Failed to create soft link /workspace: {link_process.stderr}")
                    
                    # Execute multiple cp commands in parallel
                    copy_commands = [
                        f"cp {self.workspace_dataset_path}/conda/{self.conda_dir_mapping[self.task_config.task_name]}/conda.tar /workspace && tar -xf /workspace/conda.tar -C /workspace && rm -f /workspace/conda.tar",
                        f"cp -a {self.workspace_dataset_path}/gym/{self.task_config.task_name}/task/. /workspace/task/",
                        f"cp -a {self.workspace_dataset_path}/gym/{self.task_config.task_name}/data/. /workspace/data/"
                    ]
                    
                    
                    processes = []
                    print("Start parallel copying of task files...")
                    for cmd in copy_commands:
                        processes.append(
                            subprocess.Popen(
                                cmd,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,   # Capture error information
                                text=True
                            )
                        )

                    exit_codes = []
                    for p, cmd in zip(processes, copy_commands):
                        stdout, stderr = p.communicate()
                        exit_codes.append(p.returncode)
                        if p.returncode != 0:
                            self.logger.error(f"Command execution failed: {cmd}")
                            if stdout:
                                self.logger.error(f"Standard output:\n{stdout}")
                            if stderr:
                                self.logger.error(f"Error output:\n{stderr}")

                    if any(code != 0 for code in exit_codes):
                        raise ValueError("One or more copy commands failed. Please check the log information above.")
                    
                    if "llamafactory" in self.conda_dir_mapping[self.task_config.task_name]:
                        copy_commands = [
                            f"conda run -p /workspace/conda pip uninstall -y llamafactory",
                        ]

                        processes = [subprocess.Popen(cmd, shell=True) for cmd in copy_commands]
                        
                        exit_codes = []
                        print("Start parallel llamafactory uninstall task files...")
                        for p in tqdm(processes, desc="llamafactory uninstall", unit="tasks"):
                            exit_codes.append(p.wait())

                        if any(code != 0 for code in exit_codes):
                            self.logger.error(f"llamafactory uninstall command failed. Please check permissions and paths.")
                            raise ValueError(f"llamafactory uninstall command failed. Please check permissions and paths.")
                        
                        copy_commands = [
                            f"conda run -p /workspace/conda pip install -e /workspace/task/repositories/LLaMA-Factory",
                        ]

                        processes = [subprocess.Popen(cmd, shell=True) for cmd in copy_commands]
                        
                        exit_codes = []
                        print("Start parallel llamafactory install task files...")
                        for p in tqdm(processes, desc="llamafactory install", unit="tasks"):
                            exit_codes.append(p.wait())

                        if any(code != 0 for code in exit_codes):
                            self.logger.error(f"llamafactory install command failed. Please check permissions and paths.")
                            raise ValueError(f"llamafactory install command failed. Please check permissions and paths.")
                        
                else:
                    # Use HTTP
                    use_proxy = True if computer_type == ComputerType.GPU else False
                    cmd_ops = CmdOperations(proxy_url=self.task_config.cmd_proxy_url)
                    
                    commands = [
                        f"rm -rf /workspace",
                        f"ln -s {self.actual_workspace} /workspace",
                        f"conda activate /workspace/conda && ray stop --force",
                    ]

                    for i, command in enumerate(commands):
                        run_all_cmd = False
                        sid_result = cmd_ops.create_session(computer_ip=computer_ip, http_port=computer_port, use_proxy=use_proxy)
                        if not sid_result or not sid_result.get('success'):
                            self.logger.error(f"Failed to create session: {sid_result}")
                            raise ValueError(f"Failed to create session: {sid_result}")
                        sid = sid_result.get('session_id')

                        run_all_cmd = True
                        if i == 2:
                            response = cmd_ops.run_command_for_env(command=command, session_id=sid, computer_ip=computer_ip, http_port=computer_port, use_proxy=use_proxy, wait_for_completion=False)
                            if response["success"]:
                                response = cmd_ops.session_idle(computer_ip, sid)
                                if response["success"]:
                                    while response["is_idle"]:
                                        response = cmd_ops.session_idle(computer_ip, sid)
                                        if not response["success"]:
                                            raise ValueError("cmd_ops.session_idle is wrong!") 
                                else:
                                    raise ValueError("cmd_ops.session_idle is wrong!")
                            else:
                                raise ValueError("cmd_ops.session_idle is wrong!")
                        else:
                            response = cmd_ops.run_command_for_env(command=command, session_id=sid, computer_ip=computer_ip, http_port=computer_port, use_proxy=use_proxy, wait_for_completion=True)

                        if not response['success']:
                            run_all_cmd = False
                            self.logger.error(f"Failed to execute command {command}: {response}. Next run needs to close() then reset(), ensuring no half-processed data")
                            break
                        if run_all_cmd:
                            RESET_DONE = True
                            self.task_config.computer_pool[idx].available = False
                            self.logger.info(f"Container service is ready, IP: {computer_ip}, Port: {computer_port}")
                            print(f"Container service is ready, IP: {computer_ip}, Port: {computer_port}")
                        else:
                            self.logger.error(f"Failed to connect to container service: {response}")
                            raise ValueError(f"Failed to connect to container service: {response}")
            else:
                self.logger.error(f"Total {len(self.task_config.computer_pool)} container services, currently no available containers, waiting 10s before retry")
                time.sleep(10)
        if not RESET_DONE:
            self.logger.error(f"Waited {MAX_WAIT_TIME}s, failed to connect to container service")
    
    def _reset_by_load_checkpoint(self) -> BaseObservation:
        """
        Restore environment from checkpoint
        """
        pass
    
    def close(self):
        """
        Close all GPU services
        """
        http_computer_idxs = [id for id in range(len(self.task_config.computer_pool)) if not self.task_config.computer_pool[id].available]
        if not http_computer_idxs:
            self.logger.info("No container services need to be closed")
            return
        
        self.logger.info("Closing environment")
        cmd_ops = CmdOperations(proxy_url=self.task_config.cmd_proxy_url)
        for idx in http_computer_idxs:
            computer = self.task_config.computer_pool[http_computer_idxs[idx]]
            computer_ip = computer.computer_ip
            computer_port = computer.computer_port
            computer_type = computer.computer_type
            if computer_type == ComputerType.LOCALHOST_CPU:
                rm_commands = [
                    f"rm -rf /workspace",
                ]
                processes = [subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for cmd in rm_commands]
                
                exit_codes = []
                print("Start parallel deletion of task files...")
                for p in tqdm(processes, desc="Delete task files", unit="tasks"):
                    exit_codes.append(p.wait())
                if any(code != 0 for code in exit_codes):
                    self.logger.error(f"One or more deletion commands failed. Please check permissions and paths.")
                    continue
            else:
                use_proxy = True if computer_type == ComputerType.GPU else False
                sid_result = cmd_ops.create_session(computer_ip=computer_ip, http_port=computer_port, use_proxy=use_proxy)
                if not sid_result or not sid_result.get('success'):
                    self.logger.error(f"Failed to create session: {sid_result}")
                    continue
                sid = sid_result.get('session_id')
                # response = cmd_ops.run_command_for_env(command='health', session_id=sid, computer_ip=computer_ip, http_port=computer_port, use_proxy=use_proxy, wait_for_completion=True)
                
                run_all_cmd = True
                # if response['success']:
                commands = [
                    "rm -rf /workspace",  # Delete soft link
                ]
                for command in commands:
                    response = cmd_ops.run_command_for_env(command=command, session_id=sid, computer_ip=computer_ip, http_port=computer_port, use_proxy=use_proxy, wait_for_completion=True)
                    if not response['success']:
                        self.logger.error(f"Container service IP={computer_ip}, Port={computer_port} failed to execute command {command}: {response}")
                        run_all_cmd = False
                        break
                if run_all_cmd:
                    self.task_config.computer_pool[http_computer_idxs[idx]].available = True
                    self.logger.info(f"Successfully closed container service IP={computer_ip}, Port={computer_port}")
                else:
                    self.logger.error(f"Failed to connect to container service IP={computer_ip}, Port={computer_port}: {response}")

        http_computer_failed = [computer for computer in self.task_config.computer_pool if not computer.available]
        if not http_computer_failed:
            self.logger.info(f"Successfully closed all container services")
        else:
            self.logger.error(f"{len(http_computer_failed)} container services failed to close, unclosed containers: {http_computer_failed}")