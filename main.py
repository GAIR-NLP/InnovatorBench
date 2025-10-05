#!/usr/bin/env python3
import os
import re
import asyncio
import sys
import time
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
import yaml
import concurrent.futures
import shutil
import argparse
from agents.agents import ReActAgent
from agents.context.context_managers import ReActManager
from agents.utils.checkpoint_logger import CheckpointLogger

from llm.config import AgentConfig, ModelParameters
from agents.config.types import ContextLimits
from llm.llm_client import LLMClient, infer_provider_type

from research_gym.env import BaseEnv, GZEnv
from research_gym.observation.observation import BaseObservation
from research_gym.action.action import BaseAction
from research_gym.configs.task_config import TaskConfig, ComputerConfig, ComputerType
from research_gym.schema import ActionType
from research_gym.action.system import EvalAction
from research_gym.schema.observation import ObservationType


def load_agent_config(config_path: str = "agents/config/agent_config.yaml") -> AgentConfig:
    """Loads agent configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        AgentConfig: The loaded agent configuration object.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file {config_path} does not exist.\n"
            f"Please copy agents/agent_config/agent_config_template.yaml to {config_path} and fill in your configuration."
        )
    
    with open(config_file, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
    
    # Convert to AgentConfig object
    config = AgentConfig.__new__(AgentConfig)  # Skip __init__
    config.default_provider = yaml_config.get("default_provider", "claude-3-7-sonnet-native-no-thinking")
    config.agent_type = yaml_config.get("agent_type", "react")
    config.enable_lakeview = yaml_config.get("enable_lakeview", False)
    
    # Set tool configuration
    context_tools_config = yaml_config.get("context_tools", [])
    env_tools_config = yaml_config.get("env_tools", [])
    
    # Extract tool names
    config.context_tools = [tool.get("name", "") for tool in context_tools_config if isinstance(tool, dict) and "name" in tool]
    config.env_tools = [tool.get("name", "") for tool in env_tools_config if isinstance(tool, dict) and "name" in tool]
    
    # Parse context_limits
    limits_cfg = yaml_config.get("context_limits", {}) or {}
    config.context_limits = ContextLimits(
        max_tokens=int(limits_cfg.get("max_tokens", 2000)),
        summary_threshold=int(limits_cfg.get("summary_threshold", 100000)),
        context_length=int(limits_cfg.get("context_length", 200000)),
        max_internal_action_times=int(limits_cfg.get("max_internal_action_times", -1)),
    )

    # Convert model provider configuration
    config.model_providers = {}
    for provider_name, provider_config in yaml_config.get("model_providers", {}).items():
        config.model_providers[provider_name] = ModelParameters(
            model=provider_config.get("model", ""),
            api_key=provider_config.get("api_key", ""),
            max_tokens=provider_config.get("max_tokens", 4096),
            temperature=provider_config.get("temperature", 0.7),
            top_p=provider_config.get("top_p", 1.0),
            top_k=provider_config.get("top_k", 0),
            parallel_tool_calls=provider_config.get("parallel_tool_calls", True),
            max_retries=provider_config.get("max_retries", 3),
            base_url=provider_config.get("base_url"),
            api_version=provider_config.get("api_version"),
            tool_choice=provider_config.get("tool_choice", None),
            reasoning_effort=provider_config.get("reasoning_effort", None),
            thinking=provider_config.get("thinking", False),
        )
    
    # Lakeview configuration
    if "lakeview_config" in yaml_config:
        from llm.config import LakeviewConfig
        config.lakeview_config = LakeviewConfig(
            model_provider=yaml_config["lakeview_config"].get("model_provider", config.default_provider),
            model_name=yaml_config["lakeview_config"].get("model_name", "")
        )
    else:
        config.lakeview_config = None
    
    return config

def load_task_config_from_yaml(task_name_or_path: str) -> TaskConfig:
    """Loads TaskConfig from a YAML file in research_gym/configs/tasks.

    Supports passing filename (without path or suffix) or absolute/relative path;
    When start_time is "now" or empty, automatically uses current time.

    Args:
        task_name_or_path: Task name or path to the YAML configuration file.

    Returns:
        TaskConfig: The loaded task configuration object.
    """
    tasks_dir = Path(__file__).resolve().parent / "research_gym" / "configs" / "tasks"
    path = Path(task_name_or_path)
    if not path.suffix:
        path = tasks_dir / f"{path.name}.yaml"
    elif not path.is_absolute():
        path = (tasks_dir / path.name)

    if not path.exists():
        raise FileNotFoundError(f"Task configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    def parse_start_time(value: Any) -> datetime:
        """Parses start time value into datetime object.

        Args:
            value: The start time value to parse.

        Returns:
            datetime: The parsed datetime object.
        """
        if value is None:
            return datetime.now()
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)
        if isinstance(value, str):
            val_lower = value.strip().lower()
            if val_lower in {"now", "auto", ""}:
                return datetime.now()
            try:
                return datetime.fromisoformat(value)
            except Exception:
                return datetime.now()
        return datetime.now()

    def parse_computer_type(value: Any) -> ComputerType:
        """Parses computer type value into ComputerType enum.

        Args:
            value: The computer type value to parse.

        Returns:
            ComputerType: The parsed computer type enum.
        """
        if isinstance(value, ComputerType):
            return value
        if isinstance(value, str):
            val = value.strip().lower()
            for t in ComputerType:
                if t.value == val:
                    return t
        # Default fallback to CPU
        return ComputerType.CPU

    computer_pool_data = data.get("computer_pool", []) or []
    computer_pool: List[ComputerConfig] = []
    for item in computer_pool_data:
        if not isinstance(item, dict):
            continue
        computer_pool.append(
            ComputerConfig(
                computer_ip=item.get("computer_ip", ""),
                computer_port=int(item.get("computer_port", 0) or 0),
                computer_type=parse_computer_type(item.get("computer_type", "cpu")),
                available=bool(item.get("available", True)),
            )
        )

    start_time = parse_start_time(data.get("start_time", None))

    task_config = TaskConfig(
        task_name=data.get("task_name", ""),
        workspace=data.get("workspace", ""),
        tokenizer=data.get("tokenizer", ""),
        max_working_time=float(data.get("max_working_time", 0) or 0),
        max_eval_num=int(data.get("max_eval_num", 3) or 3),
        start_time=start_time,
        max_steps=int(data.get("max_steps", 0) or 0),
        model_name=data.get("model_name"),
        checkpoint_base_path=data.get("checkpoint_base_path"),
        workspace_dataset_path=data.get("workspace_dataset_path"),
        actual_workspace=data.get("actual_workspace"),
        resume_from_path=data.get("resume_from_path"),
        save_freq=int(data.get("save_freq", -1)),
        env_vars=data.get("env_vars"),
        cmd_proxy_url=data.get("cmd_proxy_url", ""),
        computer_pool=computer_pool,
        launch_type=data.get("launch_type", ""),
        default_shell=data.get("default_shell", "/bin/bash"),
        default_http_port=data.get("default_http_port"),
        openai_api_key=data.get("openai_api_key"),
        openai_base_url=data.get("openai_base_url"),
        eval_workspace=data.get("eval_workspace", ""),
        max_eval_time=float(data.get("max_eval_time", 0) or 0),
        search_engine=data.get("search_engine", ""),
        serper_api_key=data.get("serper_api_key", ""),
        azure_bing_search_subscription_key=data.get("azure_bing_search_subscription_key", ""),
        search_max_top_k=int(data.get("search_max_top_k", 100) or 100),
        search_region=data.get("search_region", "us"),
        search_lang=data.get("search_lang", "en"),
        azure_bing_search_mkt=data.get("azure_bing_search_mkt", "en-US"),
        search_cache_dir=data.get("search_cache_dir", "./search_cache"),
        search_cache_duration_days=int(data.get("search_cache_duration_days", 1) or 1),
        web_server_host=data.get("web_server_host", "localhost"),
        web_server_port=int(data.get("web_server_port", 8124) or 8124),
        web_proxy_url=data.get("web_proxy_url", ""),
        web_cache_dir=data.get("web_cache_dir", "./web_cache"),
        web_cache_duration_days=int(data.get("web_cache_duration_days", 1) or 1),
    )
    return task_config


class Scaffold:
    """Scaffold for letting Agent run freely and record its execution process.

    1. Creates an Agent based on agent config and task config
    2. Creates an Env based on env-related config and task config
    3. Creates a ContextManager based on agent config and sets its system prompt
    4. Creates an LLMClient based on agent config
    """
    
    def __init__(self, agent_config: AgentConfig, task_config: TaskConfig, log_path: str = "agent_debug.log"):
        """Initializes the Scaffold.

        Args:
            agent_config: The agent configuration object.
            task_config: The task configuration object.
            log_path: Path to the log file. Defaults to "agent_debug.log".
        """
        self.agent_config = agent_config
        self.task_config = task_config
        self.log_path = log_path
        # Record the last sent UTC timestamp for incremental sending
        self.last_sent_timestamp = None
        # Record the last sent filename for resending
        self.last_sent_filename = None

    def _parse_node_filename(self, filename: str) -> tuple:
        """Parses node filename to extract timestamp and prefix information.

        Args:
            filename: The filename, e.g., "react_2fcf8562-7dd2-41f9-911b-c4edc5e88db8_2025-08-27T18:26:52.481996.json"

        Returns:
            tuple: (prefix, node_id, timestamp, full_filename)
        """
        if not filename.endswith('.json'):
            return None, None, None, filename
            
        # Remove .json suffix
        name_without_ext = filename[:-5]
        
        # Split by last underscore to get timestamp
        parts = name_without_ext.rsplit('_', 1)
        if len(parts) != 2:
            return None, None, None, filename
            
        prefix_part, timestamp_str = parts
        
        # Prefix part may contain multiple underscores, need to find node_id
        prefix_parts = prefix_part.split('_')
        if len(prefix_parts) >= 2:
            prefix = prefix_parts[0]  # e.g. "react"
            node_id = '_'.join(prefix_parts[1:])  # e.g. "2fcf8562-7dd2-41f9-911b-c4edc5e88db8"
        else:
            prefix = prefix_part
            node_id = None
            
        return prefix, node_id, timestamp_str, filename

    def _get_new_nodes_since_timestamp(self, since_timestamp: str = None) -> List[Dict[str, Any]]:
        """Gets all new node data since the specified timestamp.

        Args:
            since_timestamp: UTC timestamp string, if None gets all nodes.

        Returns:
            List[Dict]: List of node data dictionaries.
        """
        nodes_dir = os.path.join(self.checkpoint_path, "nodes")
        if not os.path.exists(nodes_dir):
            print(f"Node directory does not exist: {nodes_dir}")
            return []
            
        all_nodes = []
        json_files = [f for f in os.listdir(nodes_dir) if f.endswith('.json')]
        
        # Sort by filename (timestamp at end of filename, so sorted order is chronological)
        json_files.sort()
        
        for filename in json_files:
            prefix, node_id, timestamp, full_filename = self._parse_node_filename(filename)
            
            if not timestamp:
                continue
                
            # If timestamp specified, only get nodes after that timestamp
            if since_timestamp and timestamp < since_timestamp:
                continue
            print("Finding Node is in timestamp field : ",since_timestamp)
            
            file_path = os.path.join(nodes_dir, full_filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    node_content = json.load(f)
                    
                node_data = {
                    "node_id": node_id or f"node_{timestamp}",
                    "node_type": prefix or "unknown",
                    "content": node_content,
                    "timestamp": timestamp,
                    "parent_id": node_content.get("parent_id"),
                    "status": "pending",
                    "filename": full_filename  # Record filename for later processing
                }
                all_nodes.append(node_data)
                
            except Exception as e:
                print(f"Failed to read node file {full_filename}: {e}")
                continue
                
        return all_nodes

    def _save_checkpoint(self, last_action: BaseAction | None, last_observation: BaseObservation | None):
        """Saves checkpoint data to disk.

        Args:
            last_action: The last executed action.
            last_observation: The last received observation.
        """

        global_step_folder = os.path.join(self.checkpoint_path, f"global_step_{self.global_step}")
        print("global_step_folder: ", global_step_folder)

        os.makedirs(global_step_folder, exist_ok=True)
        
        def _to_serializable(value: Any) -> Any:
            """Converts a value to a serializable format.

            Args:
                value: The value to convert.

            Returns:
                Any: The serializable version of the value.
            """
            if is_dataclass(value):
                return {k: _to_serializable(v) for k, v in asdict(value).items()}
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {k: _to_serializable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple, set)):
                return [_to_serializable(v) for v in value]
            return value

        # Save task_config
        task_config_save_path = os.path.join(self.checkpoint_path, f"{self.task_config.task_name}.yaml")
        with open(task_config_save_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(_to_serializable(self.task_config), f, allow_unicode=True, sort_keys=False)

        # Save agent_config
        agent_config_save_path = os.path.join(self.checkpoint_path, f"{self.agent_config.agent_type}_agent_config.yaml")
        with open(agent_config_save_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(_to_serializable(self.agent_config), f, allow_unicode=True, sort_keys=False)

        # Save agent
        agent_params_save_path = os.path.join(global_step_folder, f"{self.agent_config.agent_type}_agent_params.json")
        agent_params = self.agent.to_dict()
        with open(agent_params_save_path, "w", encoding="utf-8") as f:
            json.dump(agent_params, f, ensure_ascii=False, indent=2)
        
        # Save context_manager
        context_manager_params_save_path = os.path.join(global_step_folder, f"{self.agent_config.agent_type}_context_manager_params.json")
        context_manager_params = self.agent.context_manager.to_dict()
        with open(context_manager_params_save_path, "w", encoding="utf-8") as f:
            json.dump(context_manager_params, f, ensure_ascii=False, indent=2)

        # Save the conversation tree of context_manager
        tree_save_path = os.path.join(global_step_folder, "tree_data.json")
        tree_data = self.agent.context_manager.root.to_dict()
        with open(tree_save_path, "w", encoding="utf-8") as f:
            json.dump(tree_data, f, ensure_ascii=False, indent=2)

        # Save last_observation
        last_observation_save_path = os.path.join(global_step_folder, "last_observation.json")
        with open(last_observation_save_path, "w", encoding="utf-8") as f:
            json.dump(self.logger.serialize_observation(last_observation), f, ensure_ascii=False, indent=2)
        
        # Save all files in workspace
        workspace_save_path = os.path.join(global_step_folder, "workspace")
        os.makedirs(workspace_save_path, exist_ok=True)
        entries = list(os.listdir("/workspace"))
        futures = []
        conda_tar_future = None

        # If conda directory exists, pack it first
        if "conda" in entries:
            shutil.make_archive("/workspace/conda", "tar", root_dir="/workspace", base_dir="conda")

        max_workers = min(8, (os.cpu_count() or 4) * 2)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for name in entries:
                if name == "conda" or name == "conda.tar":
                    # conda directory has been packed as /workspace/conda.tar, will be copied separately later
                    # and avoid duplicating with the separate conda.tar copy task
                    continue
                src = os.path.join("/workspace", name)
                dst = os.path.join(workspace_save_path, name)
                if os.path.isdir(src):
                    futures.append(executor.submit(shutil.copytree, src, dst))
                else:
                    futures.append(executor.submit(shutil.copy2, src, dst))

            # Parallel copy conda.tar (if exists)
            conda_tar_path = "/workspace/conda.tar"
            if os.path.exists(conda_tar_path):
                conda_dst = os.path.join(workspace_save_path, "conda.tar")
                conda_tar_future = executor.submit(shutil.copy, conda_tar_path, conda_dst)
                futures.append(conda_tar_future)

            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    if hasattr(self, "logger") and self.logger:
                        self.logger.log(f"Failed to copy workspace files: {e}")
                    else:
                        print(f"Failed to copy workspace files: {e}", flush=True)

        # Delete temporary /workspace/conda.tar (if created and copied successfully)
        if conda_tar_future is not None:
            try:
                conda_tar_future.result()
                if os.path.exists("/workspace/conda.tar"):
                    os.remove("/workspace/conda.tar")
            except Exception as e:
                if hasattr(self, "logger") and self.logger:
                    self.logger.log(f"Failed to delete temporary conda.tar: {e}")

        # Copy current nodes folder
        nodes_save_path = os.path.join(global_step_folder, "nodes")
        # os.makedirs(nodes_save_path, exist_ok=True)
        shutil.copytree(os.path.join(self.checkpoint_path, "nodes"), nodes_save_path)

    def _load_checkpoint(self) -> bool:
        """Attempts to restore from checkpoint:
        - Reads checkpoint
        - Rebuilds task_config / agent_config (using existing loading logic + overriding key fields)
        - Loads tree to context_manager and locates current_node
        - Rebuilds initial observation (as next step input)
        - Returns whether successful
        """
        if self.task_config.launch_type != "load_checkpoint":
            return False
        
        assert self.task_config.resume_from_path, Exception("resume_from_path of the task config is not set.")
        assert os.path.basename(self.task_config.resume_from_path).startswith('global_step_'), Exception("cannot find the checkpoint folder in the resume_from_path.")
        
        self.global_step = int(os.path.basename(self.task_config.resume_from_path).split("_")[-1])

        global_step_folder = self.task_config.resume_from_path
        print("Load checkpoint from: ", global_step_folder)
        print("Global step: ", self.global_step)

        # Read agent
        agent_params_load_path = os.path.join(global_step_folder, f"{self.agent_config.agent_type}_agent_params.json")
        with open(agent_params_load_path, "r", encoding="utf-8") as f:
            agent_params = json.load(f)
        self.agent.from_dict(agent_params)

        # Read context_manager
        context_manager_params_load_path = os.path.join(global_step_folder, f"{self.agent_config.agent_type}_context_manager_params.json")
        with open(context_manager_params_load_path, "r", encoding="utf-8") as f:
            context_manager_params = json.load(f)
        self.agent.context_manager.from_dict(context_manager_params)

        # Read tree_data
        tree_load_path = os.path.join(global_step_folder, "tree_data.json")
        self.agent.context_manager.load_from_tree(tree_load_path)
        
        # Read last_observation
        last_observation_load_path = os.path.join(global_step_folder, "last_observation.json")
        with open(last_observation_load_path, "r", encoding="utf-8") as f:
            self._last_observation = self.logger.deserialize_observation(json.load(f))
        
        # Read workspace: corresponds to save logic
        workspace_load_path = os.path.join(global_step_folder, "workspace")
        try:
            entries = list(os.listdir(workspace_load_path))
            for name in entries:
                src = os.path.join(workspace_load_path, name)
                if name == "conda.tar":
                    # Unpack to /workspace, restore to original conda directory
                    shutil.unpack_archive(src, "/workspace", format="tar")
                    continue
                dst = os.path.join("/workspace", name)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst, ignore_errors=True)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy(src, dst)
        except Exception as e:
            raise ValueError(f"Failed to restore workspace: {e}")
        
        # Copy nodes
        nodes_load_path = os.path.join(global_step_folder, "nodes")
        nodes_target_path = os.path.join(self.checkpoint_path, "nodes")
        # os.makedirs(nodes_target_path, exist_ok=True)
        shutil.copytree(nodes_load_path, nodes_target_path, dirs_exist_ok=True)
        
        return True
    
    def init(self) -> Tuple[BaseEnv, str, BaseObservation]:
        """Initializes the task environment.

        Returns:
            Tuple[BaseEnv, str, BaseObservation]: A tuple containing the environment,
                task description, and initial observation.
        """

        self.env = GZEnv(self.task_config)
        self.checkpoint_path = self.env.build_workspace_backup_folder()
        self.logger = CheckpointLogger(self.log_path, self.checkpoint_path)
        
        if self.agent_config.agent_type == "react":
            context_manager = ReActManager(self.agent_config, self.task_config)
        else:
            raise ValueError(f"Unsupported agent_type: {self.agent_config.agent_type}")

        model_parameters = self.agent_config.model_providers[self.agent_config.default_provider]
        provider_type = infer_provider_type(model_parameters)
        # Initialize LLM client
        llm_client = LLMClient(provider_type, model_parameters)

        self.agent = None
        if self.agent_config.agent_type == "react":
            self.agent = ReActAgent(
                context_manager,
                llm_client,
                self.agent_config,
                self.task_config,
            )
        else:
            raise ValueError(f"Unsupported agent_type: {self.agent_config.agent_type}")

    async def run_task(self):
        """Runs the main task execution loop.

        This method handles the complete task execution process including:
        - Initializing the environment and agent
        - Executing steps in a loop
        - Managing checkpoints and logging
        """
        # Initialize global step counter
        self.global_step = 0
        
        # Logger that supports checkpoints
        if self.task_config.launch_type != "load_checkpoint":
            # Build initial message
            init_observation = self.env.reset()
            
            if 'VIEW_HINT' in self.agent_config.env_tools:
                hint_file_path = os.path.join(self.task_config.eval_workspace, self.task_config.task_name, "hint.md")
                with open(hint_file_path, 'r', encoding='utf-8') as f:
                    hint_content = f.read()
                init_observation._message += f"\n\nHere is the hint about the task:\n\n{hint_content}\n\n" 
                
            task_description = init_observation.message
            self.agent.context_manager.root.set_task_description(task_description)

        else:
            # Load checkpoint
            self._load_checkpoint()
            init_observation = self._last_observation

        assert self.agent, Exception("Agent has not been initialized.")

        observation = init_observation
        print("init observation: ", observation.to_dict())
        
        init_step = self.global_step + 1
        for step in range(init_step, self.task_config.max_steps + 1):
            # 2. Execute one agent step and env step to get new observation
            self.global_step += 1

            start_time = self.task_config.start_time
            
            try:
                print(f"-----------------------Step {self.global_step}-----------------------")

                action, information_for_user = await self.agent.step(observation)

                # Execute env.step and get new observation
                new_observation = self.env.step(action)
                print(f"obs: {new_observation}")
                duration = (datetime.now() - start_time).total_seconds()

            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                self.logger.log(f"❌ Step execution failed (duration: {duration:.2f}s): {str(e)}")
                import traceback
                with open(self.logger.filename, 'a', encoding='utf-8') as f:
                    traceback.print_exc(file=f)
                raise

            # Fix: Check agent returned completed status
            info = self.agent.context_manager.get_information_for_user()
            is_completed = info.get("completed", False)

            # Task completion conditions: depends on agent's completed status or DONE node status
            remaining_working_time = self.task_config.max_working_time - (datetime.now() - self.agent.context_manager.start_time).total_seconds()
            if remaining_working_time <= 0 or (new_observation and new_observation.tool_name == ActionType.FINISH) or is_completed or self.agent.context_manager.current_node.node_type.value == "done" or (new_observation.observation_type == ObservationType.EVAL and new_observation.already_eval_num > self.task_config.max_eval_num):
                action = EvalAction(call_id="00000002", action_type=ActionType.EVAL)
                observation = self.env.step(action)
                print("eval observation: ", observation)
                self.logger.log(f"\n🎉 Task '{self.task_config.task_name}' completed!")
                break

            # Save a checkpoint after env.step (output status)
            if self.task_config.save_freq > 0 and self.global_step % self.task_config.save_freq == 0:
                self._save_checkpoint(
                    # step=self.global_step,
                    last_action=action,
                    last_observation=new_observation,
                )
            observation = new_observation
        else:
                self.logger.log(f"\n⚠️ Task '{self.task_config.task_name}' reached maximum steps {self.task_config.max_steps} without completion.")

        # Close environment and clear folders
        # env.close()
        # After task completion, record complete history
        await self.agent.step(observation, final_step=True) # Save eval node
        # self._save_checkpoint(
        #     # step=self.global_step,
        #     last_action=action,
        #     last_observation=new_observation,
        # )


async def main():
    """Main entry point for the agent execution.

    Parses command line arguments, loads configurations, and runs the agent task.

    Returns:
        int: Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(description="Agent execution parameters")
    parser.add_argument("--task-config", "-t", type=str, required=True, help="Task configuration file path (absolute or relative path, or task name like task_19)")
    parser.add_argument("--agent-config", "-a", type=str, required=True, help="Agent configuration file path (absolute or relative path)")
    parser.add_argument("--log-path", "-l", type=str, default=None, help="Log output file path (automatically generated unique filename if not provided)")
    args = parser.parse_args()
    
    # Match task config and agent config
    task_config_basename = os.path.basename(args.task_config)
    task_config_basename = re.search(r'(task_\d+)', task_config_basename)
    if task_config_basename:
        task_config_basename = task_config_basename.group(1)
        if task_config_basename in ["task_1","task_2","task_3","task_4","task_9","task_8"]:
            assert args.agent_config in ["agents/config/agent_config_browse.yaml"], f"{args.task_config} should use 'agents/config/agent_config_browse.yaml'."
        else:
            assert args.agent_config in ["agents/config/agent_config.yaml"], f"{args.task_config} should use 'agents/config/agent_config.yaml'."
    else:
        print("Using user's own task_config and agent_config!")

    task_config = load_task_config_from_yaml(args.task_config)
    agent_config = load_agent_config(args.agent_config)

    # Sync default_provider to TaskConfig
    task_config.model_name = agent_config.default_provider
    task_config.actual_workspace = os.path.join(task_config.actual_workspace, f"workspace_{task_config.task_name}_{task_config.model_name}_{task_config.start_time.strftime('%Y-%m-%d.%H:%M:%S')}")
    log_path = args.log_path or f"/tmp/agent_{task_config.task_name}_{task_config.start_time.strftime('%Y-%m-%d.%H:%M:%S.%f')}.log"
    print(f"🚀 Starting agent debugging, all detailed output will be recorded to log file: {log_path}")
    
    tester = Scaffold(agent_config, task_config, log_path)
    tester.init()

    print("start testing", flush=True)

    await tester.run_task()

    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 
