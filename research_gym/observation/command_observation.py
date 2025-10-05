from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Type
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum

from research_gym.base_node import BaseNode
from research_gym.schema.action import ActionType
from research_gym.schema.observation import ObservationType
from research_gym.action import BaseAction
from research_gym.observation.base_observation import BaseObservation


@dataclass
class CommandObservation(BaseObservation):
    """Base class for command operation observation results"""
    # Basic operation information
    success: bool = False
    computer_ip: Optional[str] = None
    session_id: Optional[str] = None

    def __post_init__(self):
        super().__init__()
        self._observation_type = ObservationType.COMMAND
    
    @classmethod
    def _remove_actual_workspace(cls, path: str) -> str:
        if not isinstance(path, str) or not path:
            return path
        actual = getattr(cls, 'actual_workspace', '') or ''
        if not actual:
            return path
        return path.replace(actual, '/workspace')
    
    @classmethod
    def _remove_actual_workspace_from_list(cls, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(data, List):
            return data
        for item in data:
            item['content'] = cls._remove_actual_workspace(item['content'])
        return data

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            'computer_ip': self.computer_ip,
            'session_id': self.session_id,
        })
        return {k: v for k, v in result.items() if v is not None}
    
    @classmethod
    def from_command_result(cls, result: Dict[str, Any], action: BaseAction) -> 'CommandObservation':
        assert isinstance(action.action_type, ActionType)
        action_name = action.action_type.value

        if action_name == 'create_session':
            observation = SessionCreateCommandObservation()
        elif action_name == 'list_sessions':
            observation = ListCommandObservation()
            observation.output = result.get('output', {})
        elif action_name == 'run_command':
            observation = RunCommandObservation()
            observation.command = result.get('command')
            observation.async_execution = result.get('async')
            observation.execution_time = result.get('execution_time')
            observation.output = CommandObservation._remove_actual_workspace(result.get('output', ''))
        elif action_name == 'input_in_session':
            observation = SessionInputCommandObservation()
            observation.input_text = result.get('input_text')
            observation.input_check = result.get('input_check')
        elif action_name in ['get_session_output', 'get_session_recent_output']:
            observation = SessionOutputCommandObservation()
            observation.output = CommandObservation._remove_actual_workspace(result.get('output', ''))
            observation.is_alive = result.get('is_alive')
            observation.last_activity = result.get('last_activity')
        elif action_name in ['get_session_status', 'check_session_idle']:
            observation = SessionStatusCommandObservation()
            observation.status = result.get('status')
            observation.is_idle = result.get('is_idle')
        elif action_name == 'clear_session_buffer':
            observation = SessionClearCommandObservation()
        elif action_name in ['close_session', 'close_all_sessions']:
            observation = SessionCloseCommandObservation()
            observation.num_closed_sessions = result.get('num_closed_sessions')
            observation.closed_machines = result.get('closed_machines', [])
        elif action_name == 'kill_session_processes':
            observation = SessionKillCommandObservation()
            observation.signal_used = result.get('signal_used')
            observation.killed_processes = result.get('killed_processes', [])
            observation.failed_kills = result.get('failed_kills', [])
        else:
            observation = CommandObservation()
        
        # Populate common fields from result
        observation.tool_call_id = action.call_id
        observation.tool_name = action.action_type
        
        observation.success = result.get('success', False)
        observation._message = result.get('message') if observation.success and result.get('message') else None
        observation.error_message = result.get('message') if not observation.success else None
        observation.computer_ip = result.get('computer_ip', None)
        observation.session_id = result.get('session_id', None)

        if 'timestamp' in result:
            if isinstance(result['timestamp'], (int, float)):
                observation.timestamp = datetime.fromtimestamp(result['timestamp'])
            else:
                observation._timestamp = str(result['timestamp'])

        return observation
    
    def _get_base_info(self) -> str:
        ret = super()._get_base_info()
        if self.computer_ip:
            ret += f'Computer IP: {self.computer_ip}\n'
        if self.session_id:
            ret += f'Session ID: {self.session_id}\n'
        return ret
    
    def __str__(self) -> str:
        ret = f'**CommandObservation**\n'
        ret += self._get_base_info()
        return ret

@dataclass
class SessionCreateCommandObservation(CommandObservation):
    """Observation result for 'create_session'"""
    def __post_init__(self):
        super().__post_init__()
        self._observation_type = ObservationType.COMMAND_CREATE_SESSION

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        # No unique fields to update, all handled by parent CommandObservation
        return {k: v for k, v in result.items() if v is not None}
    
    def __str__(self) -> str:
        ret = f'**SessionCreateCommandObservation**\n'
        ret += self._get_base_info()
        return ret


@dataclass
class ListCommandObservation(CommandObservation):
    """Observation result for 'list_sessions'"""
    output: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        super().__post_init__()
        self._observation_type = ObservationType.COMMAND_LIST_SESSIONS

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            'output': self.output,
        })
        return {k: v for k, v in result.items() if v is not None}
    
    def __str__(self) -> str:
        ret = f'**ListCommandObservation**\n'
        ret += self._get_base_info()
        if self.output:
            ret += f'Output: {self.output}\n'
        return ret


@dataclass
class RunCommandObservation(CommandObservation):
    """Observation result for 'run_command'"""
    command: Optional[str] = None
    async_execution: Optional[bool] = None
    execution_time: Optional[float] = None
    output: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__post_init__()
        self._observation_type = ObservationType.COMMAND_RUN

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            'command': self.command,
            'async_execution': self.async_execution,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp,
            'output': self.output,
        })
        return {k: v for k, v in result.items() if v is not None}

    def __str__(self) -> str:
        ret = f'**RunCommandObservation**\n'
        ret += self._get_base_info()
        if self.command:
            ret += f'Command: {self.command}\n'
        if self.async_execution:
            ret += f'Async execution: {self.async_execution}\n'
        if self.execution_time:
            ret += f'Execution time: {self.execution_time}\n'
        if self.output:
            ret += f'Output: \n{self.output}\n'
        return ret

@dataclass
class SessionInputCommandObservation(CommandObservation):
    """Observation result for 'input_in_session'"""
    input_text: Optional[str] = None
    input_check: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__post_init__()
        self._observation_type = ObservationType.COMMAND_INPUT_IN_SESSION

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            'input_text': self.input_text,
            'input_check': self.input_check,
        })
        return {k: v for k, v in result.items() if v is not None}

    def __str__(self) -> str:
        ret = f'**SessionInputCommandObservation**\n'
        ret += self._get_base_info()
        if self.input_text:
            ret += f'Input text: {self.input_text}\n'
        if self.input_check:
            ret += f'Input check: {self.input_check}\n'
        return ret

@dataclass
class SessionOutputCommandObservation(CommandObservation):
    """Observation result for 'get_session_output' and 'get_session_recent_output'"""
    output: Optional[Union[str, List[Dict[str, Any]]]] = None
    is_alive: Optional[bool] = None
    last_activity: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        self._observation_type = ObservationType.COMMAND_OUTPUT_SESSION

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            'output': self.output,
            'is_alive': self.is_alive,
            'last_activity': self.last_activity,
        })
        return {k: v for k, v in result.items() if v is not None}

    def __str__(self) -> str:
        ret = f'**SessionOutputCommandObservation**\n'
        ret += self._get_base_info()
        if self.output:
            ret += f'Output: \n{self.output}\n'
        if self.is_alive:
            ret += f'Is alive: {self.is_alive}\n'
        if self.last_activity:
            ret += f'Last activity: {self.last_activity}\n'
        return ret


@dataclass
class SessionStatusCommandObservation(CommandObservation):
    """Observation result for 'session_status' and 'session_idle'"""
    status: Optional[Dict[str, Any]] = None
    is_idle: Optional[bool] = None

    def __post_init__(self):
        super().__post_init__()
        self._observation_type = ObservationType.COMMAND_SESSION_STATUS

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            'status': self.status,
            'is_idle': self.is_idle,
        })
        return {k: v for k, v in result.items() if v is not None}

    def __str__(self) -> str:
        ret = f'**SessionStatusCommandObservation**\n'
        ret += self._get_base_info()
        if self.status:
            ret += f'Status: {self.status}\n'
        if self.is_idle:
            ret += f'Is idle: {self.is_idle}\n'
        return ret

@dataclass
class SessionClearCommandObservation(CommandObservation):
    """Observation result for 'clear_session_buffer'"""
    def __post_init__(self):
        super().__post_init__()
        self._observation_type = ObservationType.COMMAND_CLEAR_SESSION_BUFFER

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        # No unique fields to update, all handled by parent CommandObservation
        return {k: v for k, v in result.items() if v is not None}

    def __str__(self) -> str:
        ret = f'**SessionClearCommandObservation**\n'
        ret += self._get_base_info()
        return ret


@dataclass
class SessionCloseCommandObservation(CommandObservation):
    """Observation result for 'close_session' and 'close_all_sessions'"""
    num_closed_sessions: Optional[int] = None
    closed_machines: Optional[List[str]] = None

    def __post_init__(self):
        super().__post_init__()
        self._observation_type = ObservationType.COMMAND_CLOSE_SESSION

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            'num_closed_sessions': self.num_closed_sessions,
            'closed_machines': self.closed_machines,
        })
        return {k: v for k, v in result.items() if v is not None}

    def __str__(self) -> str:
        ret = f'**SessionCloseCommandObservation**\n'
        ret += self._get_base_info()
        if self.num_closed_sessions:
            ret += f'Number of closed sessions: {self.num_closed_sessions}\n'
        if self.closed_machines:
            ret += f'Closed machines: {self.closed_machines}\n'
        return ret


@dataclass
class SessionKillCommandObservation(CommandObservation):
    """Observation result for 'kill_session_processes'"""
    signal_used: Optional[str] = None
    killed_processes: List[Dict[str, Any]] = field(default_factory=list)
    failed_kills: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        self._observation_type = ObservationType.COMMAND_KILL_PROCESSES

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            'signal_used': self.signal_used,
            'killed_processes': self.killed_processes,
            'failed_kills': self.failed_kills,
        })
        return {k: v for k, v in result.items() if v is not None}

    def __str__(self) -> str:
        ret = f'**SessionKillCommandObservation**\n'
        ret += self._get_base_info()
        if self.signal_used:
            ret += f'Signal used: {self.signal_used}\n'
        if self.killed_processes:
            ret += f'Killed processes: {self.killed_processes}\n'
        if self.failed_kills:
            ret += f'Failed kills: {self.failed_kills}\n'
        return ret