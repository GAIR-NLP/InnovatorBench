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
class WebBrowseObservation(BaseObservation):
    """Web browsing operation observation result"""
    
    def __post_init__(self):
        super().__init__()
        self._observation_type = ObservationType.WEB_BROWSE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = super().to_dict()
        return {k: v for k, v in result.items() if v is not None}
    
    def _get_base_info(self) -> str:
        ret = super()._get_base_info()
        return ret
    
    def __str__(self) -> str:
        ret = f'**WebBrowseObservation**\n'
        ret += self._get_base_info()
        return ret
    
    @classmethod
    def from_web_result(cls, result: Dict[str, Any], action: BaseAction) -> 'WebBrowseObservation':
        """Create WebBrowseObservation from web operation result"""
        assert isinstance(action.action_type, ActionType)
        action_name = action.action_type.value

        if action_name in ["web_browse", "web_page_goto_line", "web_page_scroll_down", "web_page_scroll_up"]:
            observation = ReadWebObservation()
            observation.url = result.get("url", "")
            observation.current_line = result.get("current_line", 1)
            observation.total_lines = result.get("total_lines", 0)
            observation.start_line = result.get("start_line", 1)
            observation.end_line = result.get("end_line", 1)
            observation.header = result.get("header", "")
            observation.output = result.get("output", "")
            # observation.page_content = result.get("content", [])
        elif action_name in ["web_page_search", "web_page_search_next"]:
            observation = SearchWebObservation()
            observation.keyword = result.get("keyword", "")
            observation.search_matches = result.get("total_matches", 0)
            observation.current_match = result.get("current_match", 0)
            observation.current_line = result.get("current_line", 0)
            observation.output = result.get("output", "")
            observation.context = result.get("context", [])
        elif action_name == "web_page_get_links":
            observation = ListLinksObservation()
            observation.links = result.get("links", [])
            observation.page_number = result.get("page_number", 1)
            observation.total_pages = result.get("total_pages", 1)
            observation.page_size = result.get("page_size", 10)
        else:
            observation = cls()


        observation.tool_call_id = action.call_id
        observation.tool_name = action.action_type
        observation.success = result.get("success")

        if observation.success:
            observation._message = result.get("message")
        else:
            observation.error_message = result.get("message")

        return observation


@dataclass
class ReadWebObservation(WebBrowseObservation):
    """goto_line, goto, scroll_down, scroll_up Web-read observation result"""
    url: str = ""                       # Current URL
    current_line: int = 1               # Current line number
    total_lines: int = 0                # Total lines
    start_line: int = 1
    end_line: int = 1
    header: str = ""
    output: str = ""
    page_content: List[str] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        self._observation_type = ObservationType.WEB_BROWSE_READ

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            'start_line': self.start_line,
            'end_line': self.end_line,
            'header': self.header,
            'output': self.output,
            # 'page_content': self.page_content,
        })
        return {k: v for k, v in result.items() if v is not None}

    def __str__(self) -> str:
        ret = f'**ReadWebObservation**\n'
        ret += self._get_base_info()
        if self.url:
            ret += f'URL: {self.url}\n'
        if self.current_line:
            ret += f'Current line: {self.current_line}\n'
        if self.total_lines:
            ret += f'Total lines: {self.total_lines}\n'
        if self.start_line:
            ret += f'Start line: {self.start_line}\n'
        if self.end_line:
            ret += f'End line: {self.end_line}\n'
        if self.header:
            ret += f'Header: {self.header}\n'
        if self.output:
            ret += f'Output: \n{self.output}\n'
        if self.page_content:
            ret += f'Page content: \n{self.page_content}\n'
        return ret


@dataclass
class SearchWebObservation(WebBrowseObservation):
    """search, search_next Web-search observation result"""
    keyword: str = ""
    search_matches: int = 0
    current_match: int = 0
    current_line: int = 0
    context: List[str] = field(default_factory=list)
    output: str = ""

    def __post_init__(self):
        super().__post_init__()
        self._observation_type = ObservationType.WEB_BROWSE_SEARCH

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            'keyword': self.keyword,
            'search_matches': self.search_matches,
            'current_match': self.current_match,
            'current_line': self.current_line,
            'context': self.context,
            'output': self.output,
        })
        return {k: v for k, v in result.items() if v is not None}

    def __str__(self) -> str:
        ret = f'**SearchWebObservation**\n'
        ret += self._get_base_info()
        if self.keyword:
            ret += f'Keyword: {self.keyword}\n'
        if self.search_matches:
            ret += f'Search matches: {self.search_matches}\n'
        if self.current_match:
            ret += f'Current match: {self.current_match}\n'
        if self.current_line:
            ret += f'Current line: {self.current_line}\n'
        if self.context:
            ret += f'Context: \n{self.context}\n'
        if self.output:
            ret += f'Output: \n{self.output}\n'
        return ret

@dataclass
class ListLinksObservation(WebBrowseObservation):
    """get_links Web-links observation result"""
    total_links: int = 0                # Total links
    page_number: int = 0
    total_pages: int = 0
    page_size: int = 0
    links: List[Dict[str, Any]] = field(default_factory=list)
    output: str = ""

    def __post_init__(self):
        super().__post_init__()
        self._observation_type = ObservationType.WEB_BROWSE_LINKS

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            'total_links': self.total_links,
            'page_number': self.page_number,
            'total_pages': self.total_pages,
            'page_size': self.page_size,
            'links': self.links,
            'output': self.output,
        })
        return {k: v for k, v in result.items() if v is not None}

    def __str__(self) -> str:
        ret = f'**ListLinksObservation**\n'
        ret += self._get_base_info()
        if self.total_links:
            ret += f'Total links: {self.total_links}\n'
        if self.page_number:
            ret += f'Page number: {self.page_number}\n'
        if self.total_pages:
            ret += f'Total pages: {self.total_pages}\n'
        if self.page_size:
            ret += f'Page size: {self.page_size}\n'
        if self.links:
            ret += f'Links: \n{self.links}\n'
        if self.output:
            ret += f'Output: \n{self.output}\n'
        return ret