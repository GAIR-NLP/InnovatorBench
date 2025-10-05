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
class SearchObservation(BaseObservation):
    """Search operation observation result"""
    query: str = ""                     # Search query
    requested_top_k: int = 0            # Requested top_k
    total_results: int = 0              # Total results
    search_results: List[Dict[str, Any]] = field(default_factory=list)  # Search results
    from_cache: bool = False            # Whether from cache

    # Error information
    error_code: Optional[str] = None
    
    def __post_init__(self):
        super().__init__()
        self._observation_type = ObservationType.WEB_SEARCH
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = super().to_dict()
        result.update({
            'query': self.query,
            'requested_top_k': self.requested_top_k,
            'total_results': self.total_results,
            'search_results': self.search_results,
            'from_cache': self.from_cache,
            'error_code': self.error_code,
        })
        return {k: v for k, v in result.items() if v is not None}
    
    def _get_base_info(self) -> str:
        ret = super()._get_base_info()
        if self.query:
            ret += f'Query: {self.query}\n'
        if self.requested_top_k:
            ret += f'Requested top k: {self.requested_top_k}\n'
        if self.total_results:
            ret += f'Total results: {self.total_results}\n'
        if self.search_results:
            ret += f'Search results: {self.search_results}\n'
        if self.from_cache:
            ret += f'From cache: {self.from_cache}\n'
        if self.error_code:
            ret += f'Error code: {self.error_code}\n'
        return ret

    @classmethod
    def from_search_result(cls, result: Dict[str, Any], action: BaseAction) -> 'SearchObservation':
        """Create SearchObservation from search result"""
        assert isinstance(action.action_type, ActionType)
        observation = cls()
        observation.tool_call_id = action.call_id
        observation.tool_name = action.action_type
        
        observation.success = result.get("success")
        if observation.success:
            observation._message = result.get("message")
            observation.query = result.get("query")
            observation.requested_top_k = result.get("requested_top_k")
            observation.total_results = result.get("total_results")
            observation.search_results = result.get("search_results")
            observation.from_cache = result.get("from_cache")
        else:
            observation.error_message = result.get("message")
            observation.error_code = result.get("error_code")
        return observation
    
    def __str__(self) -> str:
        ret = f'**SearchObservation**\n'
        ret += self._get_base_info()
        return ret