from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Type
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum

from research_gym.base_node import BaseNode
from research_gym.schema.action import ActionType
from research_gym.schema.observation import ObservationType
from research_gym.action import BaseAction
from research_gym.observation.command_observation import CommandObservation
from research_gym.observation.file_observation import FileObservation
from research_gym.observation.parse_observation import ParseObservation
from research_gym.observation.eval_observation import EvalObservation
from research_gym.observation.search_observation import SearchObservation
from research_gym.observation.web_browse_observation import WebBrowseObservation
from research_gym.observation.base_observation import BaseObservation


# Observation result factory
class ObservationFactory:
    """Observation result factory class - uses factory pattern to create specific observation results"""
    
    @staticmethod
    def create_command_observation(result: Dict[str, Any], action: BaseAction) -> CommandObservation:
        """Create command observation result"""
        return CommandObservation.from_command_result(result, action)
    
    @staticmethod
    def create_file_observation(result: Dict[str, Any], action: BaseAction) -> FileObservation:
        """Create file observation result"""
        return FileObservation.from_file_result(result, action)
    
    @staticmethod
    def create_parse_observation(result: Union[str, Dict[str, Any]], action: BaseAction) -> ParseObservation:
        """Create parse observation result"""
        return ParseObservation.from_parse_result(result, action)
    
    @staticmethod
    def create_eval_observation(result: Dict[str, Any], action: BaseAction) -> EvalObservation:
        """Create evaluation observation result"""
        return EvalObservation.from_eval_result(result, action)
    
    @staticmethod
    def create_search_observation(result: Dict[str, Any], action: BaseAction) -> SearchObservation:
        """Create search observation result"""
        return SearchObservation.from_search_result(result, action)
    
    @staticmethod
    def create_web_browse_observation(result: Dict[str, Any], action: BaseAction) -> WebBrowseObservation:
        """Create web browse observation result"""
        return WebBrowseObservation.from_web_result(result, action)
    
    @staticmethod
    def create_base_observation(result: Dict[str, Any], action: BaseAction) -> BaseObservation:
        """Create base observation result"""
        return BaseObservation.from_result(result, action)
    
    @staticmethod
    def create_observation_by_type(
        observation_type: ObservationType, 
        result: Union[str, Dict[str, Any]], 
        action: BaseAction
    ) -> BaseObservation:
        """Create observation result by type"""
        if observation_type in [ObservationType.COMMAND, ObservationType.COMMAND_CREATE_SESSION, ObservationType.COMMAND_LIST_SESSIONS, ObservationType.COMMAND_RUN, ObservationType.COMMAND_INPUT_IN_SESSION, ObservationType.COMMAND_OUTPUT_SESSION, ObservationType.COMMAND_SESSION_STATUS, ObservationType.COMMAND_CLEAR_SESSION_BUFFER, ObservationType.COMMAND_CLOSE_SESSION, ObservationType.COMMAND_KILL_PROCESSES]:
            return ObservationFactory.create_command_observation(result, action)
        elif observation_type in [ObservationType.FILE, ObservationType.FILE_READ, ObservationType.FILE_WRITE, ObservationType.FILE_ERROR, ObservationType.FILE_INFO, ObservationType.FILE_LIST, ObservationType.FILE_SEARCH]:
            return ObservationFactory.create_file_observation(result, action)
        elif observation_type == ObservationType.PARSE:
            return ObservationFactory.create_parse_observation(result, action)
        elif observation_type == ObservationType.EVAL:
            return ObservationFactory.create_eval_observation(result, action)
        elif observation_type == ObservationType.WEB_SEARCH:
            return ObservationFactory.create_search_observation(result, action)
        elif observation_type in [ObservationType.WEB_BROWSE, ObservationType.WEB_BROWSE_READ, ObservationType.WEB_BROWSE_SEARCH, ObservationType.WEB_BROWSE_LINKS]:
            return ObservationFactory.create_web_browse_observation(result, action)
        else:
            return ObservationFactory.create_base_observation(result, action)