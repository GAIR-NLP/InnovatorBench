import os
import json
import tempfile
from datetime import datetime
from typing import Any, Dict, Optional

from research_gym.action.action import BaseAction
from research_gym.action.action_type_mapping import get_action_class
from research_gym.observation.observation import BaseObservation
from research_gym.configs.task_config import TaskConfig, ComputerType
from research_gym.schema.action import ActionType


class CheckpointLogger:
	"""Logger responsible for log recording and checkpoint saving/restoring.

	- Provide simple log interface (compatible with FileLogger usage)
	- Provide save_checkpoint/load_checkpoint interface, atomic writing, prevent partial write corruption
	- Don't care how to construct checkpoint, only responsible for persistence and reading
	"""

	def __init__(self, log_file: str = "agent_debug.log", checkpoint_path: Optional[str] = None):
		self.filename = log_file
		self.checkpoint_path = checkpoint_path
		# Initialize log file (overwrite write, indicates new session)
		try:
			with open(self.filename, 'w', encoding='utf-8') as f:
				f.write(f"Log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
		except IOError as e:
			print(f"Error: Could not write to log file {self.filename}. Error: {e}")

	def log(self, message: str):
		"""Append a message to log file"""
		try:
			with open(self.filename, 'a', encoding='utf-8') as f:
				f.write(message + '\n')
		except IOError as e:
			print(f"Error: Could not write to log file {self.filename}. Error: {e}")

	def serialize_action(self, action: Optional[BaseAction]) -> Optional[Dict[str, Any]]:
		if action is None:
			return None
		try:
			data = {k: v for k, v in vars(action).items() if isinstance(v, (str, int, float, bool, list, dict))}
			data.update({
				"action_type": getattr(action, "action_type", None).value if getattr(action, "action_type", None) else None,
				"call_id": getattr(action, "call_id", None)
			})
			return data
		except Exception:
			return {"repr": str(action)}

	def serialize_task_config(self, task_config: Optional[TaskConfig]) -> Optional[Dict[str, Any]]:
		if task_config is None:
			return None
		try:
			data = {
				"task_name": task_config.task_name,
				"workspace": task_config.workspace,
				"tokenizer": task_config.tokenizer,
				"max_working_time": task_config.max_working_time,
				"start_time": task_config.start_time.isoformat() if isinstance(task_config.start_time, datetime) else str(task_config.start_time),
				"max_steps": task_config.max_steps,
				"checkpoint_base_path": task_config.checkpoint_base_path,
				"actual_workspace": task_config.actual_workspace,
				"env_vars": task_config.env_vars,
				"cmd_proxy_url": task_config.cmd_proxy_url,
				"computer_pool": [
					{
						"computer_ip": c.computer_ip,
						"computer_port": c.computer_port,
						"computer_type": c.computer_type.value if hasattr(c.computer_type, "value") else str(c.computer_type),
						"available": c.available,
					} for c in (task_config.computer_pool or [])
				],
				"launch_type": task_config.launch_type,
				"default_shell": task_config.default_shell,
				"default_http_port": task_config.default_http_port,
				"openai_api_key": task_config.openai_api_key,
				"openai_base_url": task_config.openai_base_url,
				"eval_workspace": task_config.eval_workspace,
				"max_eval_time": task_config.max_eval_time,
				"search_engine": task_config.search_engine,
				"serper_api_key": task_config.serper_api_key,
				"azure_bing_search_subscription_key": task_config.azure_bing_search_subscription_key,
				"search_max_top_k": task_config.search_max_top_k,
				"search_region": task_config.search_region,
				"search_lang": task_config.search_lang,
				"azure_bing_search_mkt": task_config.azure_bing_search_mkt,
				"search_cache_dir": task_config.search_cache_dir,
				"search_cache_duration_days": task_config.search_cache_duration_days,
				"web_server_host": task_config.web_server_host,
				"web_server_port": task_config.web_server_port,
				"web_proxy_url": task_config.web_proxy_url,
				"web_cache_dir": task_config.web_cache_dir,
				"web_cache_duration_days": task_config.web_cache_duration_days,
			}
			return data
		except Exception:
			return {"task_name": getattr(task_config, "task_name", None), "workspace": getattr(task_config, "workspace", None)}

	def serialize_observation(self, obs: Optional[BaseObservation]) -> Optional[Dict[str, Any]]:
		return obs.to_dict() if obs else None

	def deserialize_observation(self, d: Dict[str, Any]) -> BaseObservation:
		obs = BaseObservation()
		obs.success = bool(d.get("success", True))
		msg = d.get("message")
		if obs.success:
			obs._message = msg
		else:
			obs.error_message = msg
		ts = d.get("timestamp")
		try:
			if isinstance(ts, (int, float)):
				obs.timestamp = datetime.fromtimestamp(ts)
			elif isinstance(ts, str):
				obs.timestamp = datetime.fromisoformat(ts)
		except Exception:
			pass
		obs.tool_call_id = d.get("tool_call_id")
		obs.tool_name = d.get("tool_name")
		obs._source = d.get("source")
		return obs
	
	def deserialize_action(self, d: Dict[str, Any]) -> BaseAction:
		action_type = d["action_type"]
		d.pop("action_type")
		action = get_action_class(ActionType(action_type))(**d)
		return action