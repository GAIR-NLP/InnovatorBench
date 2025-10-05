from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Type
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum

from research_gym.base_node import BaseNode
from research_gym.schema.action import ActionType
from research_gym.schema.observation import ObservationType
from research_gym.action import BaseAction


@dataclass
class BaseObservation(BaseNode):
    """Base observation class"""
    tool_call_id: str | None = None
    tool_name: ActionType | None = None
    success: bool = True
    
    # Error message
    error_message: Optional[str] = None

    actual_workspace: str = ""

    def __init__(self):
        super().__init__()
        self._observation_type = ObservationType.BASE
    
    @property
    def observation_type(self) -> ObservationType:
        """Get observation result type"""
        return self._observation_type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = {
            'success': self.success,
            'message': self.message,
            'error_message': self.error_message,
            'observation_type': self.observation_type.value,
            'timestamp': self.timestamp,
            'source': self.source.value if self.source is not None else None,
        }
        
        # Add BaseNode attributes
        if self.id != BaseNode.INVALID_ID:
            result['id'] = self.id
        if self.cause is not None:
            result['cause'] = self.cause
        if self.timeout is not None:
            result['timeout'] = self.timeout
            
        return {k: v for k, v in result.items() if v is not None and v != ""}
    
    @classmethod
    def from_result(cls, result: Dict[str, Any], action: BaseAction) -> 'BaseObservation':
        """Create BaseObservation from result"""
        assert type(action.action_type) == ActionType
        observation = cls()
        observation.tool_call_id = action.call_id
        observation.tool_name = action.action_type
        observation.success = result.get('success', False)
        observation._source = result.get('source', None)
        if observation.success:
            observation._message = result.get('message')
        else:
            observation.error_message = result.get('message')

        # Set timestamp
        if 'timestamp' in result:
            if isinstance(result['timestamp'], (int, float)):
                observation.timestamp = datetime.fromtimestamp(result['timestamp'])
            else:
                observation._timestamp = str(result['timestamp'])
        
        return observation
    
    def _get_base_info(self) -> str:
        ret = ""
        if self.success:
            ret += f'Success: {self.success}\n'
        if self.tool_call_id:
            ret += f'Tool call ID: {self.tool_call_id}\n'
        if self.tool_name:
            ret += f'Tool name: {self.tool_name}\n'
        if self.message:
            ret += f'Message: {self.message}\n'
        if self.error_message:
            ret += f'Error message: {self.error_message}\n'
        if self.timestamp:
            ret += f'Timestamp: {self.timestamp}\n'
        return ret
    
    def __str__(self) -> str:
        ret = f'**BaseObservation**\n'
        ret += self._get_base_info()
        return ret