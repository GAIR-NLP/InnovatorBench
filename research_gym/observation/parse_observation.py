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
class ParseObservation(BaseObservation):
    """Parse operation observation result"""
    
    # Basic information
    success: bool = False
    parse_type: str = "unknown"  # pdf, docx, latex, audio, image, video, pptx
    
    # Parse result
    parsed_content: str = ""
    save_path: str = ""
    
    # Processing information
    model_used: Optional[str] = None
    
    def __post_init__(self):
        super().__init__()
        self._observation_type = ObservationType.PARSE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = super().to_dict()
        result.update({
            'parse_type': self.parse_type,
            'parsed_content': self.parsed_content,
            'save_path': self.save_path,
            'model_used': self.model_used,
        })
        return {k: v for k, v in result.items() if v is not None}
    
    def _get_base_info(self) -> str:
        ret = super()._get_base_info()
        if self.parse_type:
            ret += f'Parse type: {self.parse_type}\n'
        if self.parsed_content:
            ret += f'Parsed content: \n{self.parsed_content}\n'
        if self.save_path:
            ret += f'Save path: {self.save_path}\n'
        if self.model_used:
            ret += f'Model used: {self.model_used}\n'
        return ret
    
    def __str__(self) -> str:
        ret = f'**ParseObservation**\n'
        ret += self._get_base_info()
        return ret
    
    @classmethod
    def from_parse_result(cls, result: Dict[str, Any], action: BaseAction) -> 'ParseObservation':
        """Create ParseObservation from parser_operations.py return result"""
        observation = cls()
        
        observation.tool_call_id = action.call_id
        assert isinstance(action.action_type, ActionType)
        observation.tool_name = action.action_type
        
        # Process according to the return format of parser_operations.py
        observation.success = result.get('success', False)

        if observation.success:
            observation._message = result.get('message')
            observation.save_path = result.get('save_path', '')
        else:
            observation.error_message = result.get('message')
        
        # Get information from action
        if action:
            
            # Infer parse type
            action_name = action.__class__.__name__.lower()
            if 'pdf' in action_name:
                observation.parse_type = 'pdf'
            elif 'docx' in action_name:
                observation.parse_type = 'docx'
            elif 'latex' in action_name:
                observation.parse_type = 'latex'
            elif 'audio' in action_name:
                observation.parse_type = 'audio'
                observation.model_used = getattr(action, 'model', 'whisper-1')
            elif 'image' in action_name:
                observation.parse_type = 'image'
            elif 'video' in action_name:
                observation.parse_type = 'video'
            elif 'pptx' in action_name:
                observation.parse_type = 'pptx'
        
        return observation
    
    def is_successful(self) -> bool:
        """Check if parsing was successful"""
        return self.success and self.error_message is None
    
    def has_content(self) -> bool:
        """Check if there is parsed content"""
        return bool(self.parsed_content.strip())
    
    def get_content_preview(self, max_length: int = 200) -> str:
        """Get content preview"""
        if not self.parsed_content:
            return ""
        
        content = self.parsed_content.strip()
        if len(content) <= max_length:
            return content
        
        return content[:max_length] + "..."