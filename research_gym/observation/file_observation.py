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
class FileObservation(BaseObservation):
    """File operation observation result"""
    
    # Basic operation information
    success: bool = False
    file_path: Optional[str] = None
    dir_path: Optional[str] = None
    
    def __post_init__(self):
        super().__init__()
        self._observation_type = ObservationType.FILE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = super().to_dict()
        result.update({
            'file_path': self.file_path,
            'dir_path': self.dir_path,
        })
        return {k: v for k, v in result.items() if v is not None}
    
    def _get_base_info(self) -> str:
        ret = super()._get_base_info()
        if self.file_path:
            ret += f'File path: {self.file_path}\n'
        if self.dir_path:
            ret += f'Dir path: {self.dir_path}\n'
        return ret
    
    def __str__(self) -> str:
        ret = f'**FileObservation**\n'
        ret += self._get_base_info()
        return ret

    @classmethod
    def from_file_result(cls, result: Dict[str, Any], action: BaseAction) -> 'FileObservation':
        """Create FileObservation from file operation result"""
        action_name = action.action_type.value
        
        # First check if the operation was successful
        if not result.get('success', False):
            observation = ErrorObservation()
            observation.error_message = result.get('message')
        else: 
            # Determine which subclass to instantiate
            if action_name in ['open_file', 'file_goto_line', 'file_scroll_down', 'file_scroll_up']:
                observation = ReadFileObservation()
                # observation.content_lines = result.get('content', [])
                content_lines = result.get('content', [])
                observation.output = FileObservation._remove_actual_workspace(result.get('output', ''))
                observation.current_line = result.get('current_line', 1)
                observation.total_lines = result.get('total_lines', len(content_lines))
                observation.start_line = result.get('start_line', 1)
                observation.end_line = result.get('end_line', observation.total_lines)
                observation.file_exists = result.get('success', False)
            elif action_name in ['create_file', 'edit_file']:
                observation = WriteFileObservation()
                observation.removed_context = result.get('removed_context',"")
                observation.added_context = result.get('added_context',"")
                # observation.new_total_lines = result.get('new_total_lines')
            elif action_name in ['search_dir', 'search_file', 'find_file']:
                observation = SearchFileDirObservation()
                observation.search_matches = result.get('matches', [])
                observation.search_term = result.get('search_term') or getattr(action, 'search_term', None)
                observation.num_matches = result.get('num_matches', 0)
                observation.num_files = result.get('num_files')
                observation.output = FileObservation._remove_actual_workspace(result.get('output', ''))
            elif action_name == 'list_files':
                observation = ListFileObservation()
                observation.list_directories = result.get('list_directories', [])
                observation.list_files = result.get('list_files', [])
                observation.total_items = result.get('total_items', 0)
            elif action_name == 'get_file_info':
                observation = FileInfoObservation()
                observation.current_line = result.get('current_line', 1)
                observation.total_lines = result.get('total_lines', 0)
                observation.window_size = result.get('window_size', 100)
            else:
                observation = FileObservation()
            
            observation._message = result.get('message') if result.get('success', False) and result.get('message') else None

        observation.tool_call_id = action.call_id
        assert isinstance(action.action_type, ActionType)
        observation.tool_name = action.action_type
        
        # Basic information
        observation.success = result.get('success', False)
        observation.file_path = result.get('file_path', None)
        observation.dir_path = result.get('dir_path', None)
        
        return observation
    
    @classmethod
    def _remove_actual_workspace(cls, path: str) -> str:
        if not isinstance(path, str) or not path:
            return path
        actual = getattr(cls, 'actual_workspace', '') or ''
        if not actual:
            return path
        return path.replace(actual, '/workspace')

    def get_content_text(self) -> str:
        """Get file content text"""
        if hasattr(self, 'content_lines') and isinstance(self.content_lines, list):
            return '\n'.join(self.content_lines)
        return ''
    
    def is_successful(self) -> bool:
        """Check if operation was successful"""
        return self.success and self.error_message is None
    
    def has_content(self) -> bool:
        """Check if there is content"""
        return len(self.content_lines) > 0 if hasattr(self, 'content_lines') else False
    
    def get_search_results_summary(self) -> str:
        """Get search results summary"""
        if not self.search_matches:
            return "No matches found"
        return f"Found {len(self.search_matches)} matches for '{self.search_term}'"


@dataclass
class ReadFileObservation(FileObservation):
    """Observation for reading file content (open, goto, scroll)."""
    # content_lines: List[str] = field(default_factory=list)
    output: str = ''
    current_line: int = 1
    total_lines: int = 0
    start_line: int = 1
    end_line: int = 1
    file_exists: bool = False

    def __post_init__(self):
        super().__post_init__()
        self._observation_type = ObservationType.FILE_READ

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            # 'content_lines': self.content_lines,
            'output': self.output,
            'current_line': self.current_line,
            'total_lines': self.total_lines,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'file_exists': self.file_exists,
        })
        return {k: v for k, v in result.items() if v is not None}

    def __str__(self) -> str:
        ret = f'**ReadFileObservation**\n'
        ret += self._get_base_info()
        if self.output:
            ret += f'Output: \n{self.output}\n'
        if self.current_line:
            ret += f'Current line: {self.current_line}\n'
        if self.total_lines:
            ret += f'Total lines: {self.total_lines}\n'
        if self.start_line:
            ret += f'Start line: {self.start_line}\n'
        if self.end_line:
            ret += f'End line: {self.end_line}\n'
        if self.file_exists:
            ret += f'File exists: {self.file_exists}\n'
        return ret

@dataclass
class WriteFileObservation(FileObservation):
    """Observation for writing/editing a file."""
    new_total_lines: Optional[int] = None
    removed_context: Optional[str] = None
    added_context: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        self._observation_type = ObservationType.FILE_WRITE

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            'removed_context': self.removed_context,
            'added_context': self.added_context,
        })
        # result.update({
        #     'new_total_lines': self.new_total_lines,
        # })
        return {k: v for k, v in result.items() if v is not None}

    def __str__(self) -> str:
        ret = f'**WriteFileObservation**\n'
        ret += self._get_base_info()
        if self.new_total_lines:
            ret += f'New total lines: {self.new_total_lines}\n'
        return ret


@dataclass
class SearchFileDirObservation(FileObservation):
    """Observation for searching in files or directories."""
    search_matches: List[Dict[str, Any]] = field(default_factory=list)
    search_term: Optional[str] = None
    num_matches: int = 0
    num_files: Optional[int] = None
    output: str = ''
    # content_lines: List[str] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        self._observation_type = ObservationType.FILE_SEARCH

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            'search_matches': self.search_matches,
            'search_term': self.search_term,
            'num_matches': self.num_matches,
            'num_files': self.num_files,
            'output': self.output,
        })
        return {k: v for k, v in result.items() if v is not None}

    def __str__(self) -> str:
        ret = f'**SearchFileDirObservation**\n'
        ret += self._get_base_info()
        if self.search_matches:
            ret += f'Search matches: {self.search_matches}\n'
        if self.search_term:
            ret += f'Search term: {self.search_term}\n'
        if self.num_matches:
            ret += f'Number of matches: {self.num_matches}\n'
        if self.num_files:
            ret += f'Number of files: {self.num_files}\n'
        if self.output:
            ret += f'Output: \n{self.output}\n'
        return ret


@dataclass
class ListFileObservation(FileObservation):
    """Observation for listing directory contents."""
    list_directories: List[str] = field(default_factory=list)
    list_files: List[str] = field(default_factory=list)
    total_items: int = 0

    def __post_init__(self):
        super().__post_init__()
        self._observation_type = ObservationType.FILE_LIST

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            'list_directories': self.list_directories,
            'list_files': self.list_files,
            'total_items': self.total_items,
        })
        return {k: v for k, v in result.items() if v is not None}

    def __str__(self) -> str:
        ret = f'**ListFileObservation**\n'
        ret += self._get_base_info()
        if self.list_directories:
            ret += f'List directories: {self.list_directories}\n'
        if self.list_files:
            ret += f'List files: {self.list_files}\n'
        if self.total_items:
            ret += f'Total items: {self.total_items}\n'
        return ret

@dataclass
class FileInfoObservation(FileObservation):
    """Observation for getting file information."""
    current_line: int = 1
    total_lines: int = 0
    window_size: int = 100

    def __post_init__(self):
        super().__post_init__()
        self._observation_type = ObservationType.FILE_INFO

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            'current_line': self.current_line,
            'total_lines': self.total_lines,
            'window_size': self.window_size,
        })
        return {k: v for k, v in result.items() if v is not None}

    def __str__(self) -> str:
        ret = f'**FileInfoObservation**\n'
        ret += self._get_base_info()
        if self.current_line:
            ret += f'Current line: {self.current_line}\n'
        if self.total_lines:
            ret += f'Total lines: {self.total_lines}\n'
        if self.window_size:
            ret += f'Window size: {self.window_size}\n'
        return ret

@dataclass
class ErrorObservation(FileObservation):
    """Observation for a failed file operation."""

    def __post_init__(self):
        super().__post_init__()
        self._observation_type = ObservationType.FILE_ERROR
        self.success = False

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        return {k: v for k, v in result.items() if v is not None}

    def __str__(self) -> str:
        ret = f'**ErrorObservation**\n'
        ret += self._get_base_info()
        return ret