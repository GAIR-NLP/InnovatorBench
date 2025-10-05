from dataclasses import dataclass, field
from typing import ClassVar, Optional, List

from research_gym.schema.action import ActionType
from research_gym.action.action import BaseAction, ActionSecurityRisk


# wuyz: Here we temporarily don't consider openhands-aci edit mode, only consider llm-based edit mode

@dataclass
class FileEditAction(BaseAction):
    """Edits a file using various commands including view, create, str_replace, insert, and undo_edit.

    This class supports two main modes of operation:
    1. LLM-based editing

    Attributes:
        path (str): The path to the file being edited.
        LLM-based editing arguments:
            content (str): The content to be written or edited in the file (used in LLM-based editing and 'write' command).
            start (int): The starting line for editing (1-indexed, inclusive). Default is 1.
            end (int): The ending line for editing (1-indexed, inclusive). Default is -1 (end of file).
            thought (str): The reasoning behind the edit action.
            action (str): The type of action being performed (always ActionType.EDIT).
        runnable (bool): Indicates if the action can be executed (always True).
        security_risk (ActionSecurityRisk | None): Indicates any security risks associated with the action.

    Usage:
        - For LLM-based editing: Use path, content, start, and end attributes.

    Note:
        - If start is set to -1 in LLM-based editing, the content will be appended to the file.
        - The 'write' command behaves similarly to LLM-based editing, using content, start, and end attributes.
    """

    action_type: ActionType = ActionType.EDIT
    path: str = field(
        default='',
        metadata={
            'tool_param': True,
            'description': 'The path to the file to edit.',
            'required': True
        }
    )

    start: int = field(
        default=1,
        metadata={
            'tool_param': True,
            'description': 'The starting line to be edited. (including)',
            'required': True
        }
    )
    end: int = field(
        default=-1,
        metadata={
            'tool_param': True,
            'description': 'The ending line to be edited. (including)',
            'required': True
        }
    )
    # LLM-based editing arguments
    content: str = field(
        default='',
        metadata={
            'tool_param': True,
            'description': 'The content to be written or edited in the file. It will replace the content between `start` and `end` lines.' ,
            'required': True
        }
    )
    

    # Shared arguments
    thought: str = ''
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None
    description: str = "Edit a file given path. The file's [start,end] lines will be edit to the content. Remember this edit will change the file's line-linenumber index, so do not edit consecutively until you use `read_file` tools to read the new file version"

    def __str__(self) -> str:
        ret = '**FileEditAction**\n'
        if self.path:
            ret += f'Path: [{self.path}]\n'
        if self.thought:
            ret += f'Thought: {self.thought}\n'
        if self.start or self.end:
            ret += f'Range: [L{self.start}:L{self.end}]\n'
        if self.content:
            ret += f'Content:\n```\n{self.content}\n```\n'
        return ret


@dataclass
class OpenFileAction(BaseAction):
    """Opens a file and displays its content around a specific line.
    
    Attributes:
        path (str): The path to the file to open.
        line_number (int): The line number to focus on (1-indexed). Default is 1.
        context_lines (Optional[int]): Number of lines to show as context. Default is None (uses default window size).
        thought (str): The reasoning behind the open action.
    """

    path: str = field(
        default='',
        metadata={
            'tool_param': True,
            'description': 'The path to the file to open.',
            'required': True
        }
    )
    line_number: int = field(
        default=1,
        metadata={
            'tool_param': True,
            'description': 'The line number to focus on (1-indexed).',
            'required': True
        }
    )
    context_lines: Optional[int] = field(
        default=None,
        metadata={
            'tool_param': True,
            'description': 'Number of lines to show as context. Default is None (uses default window size).',
            'required': True
        }
    )
    thought: str = ''
    action_type: ActionType = ActionType.OPEN_FILE
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None
    description: str = "Open a file and display its content around a specific line. The environment will cache the file content for another file action to use until perform next open_file action."

    @property
    def message(self) -> str:
        return f'Opening file: {self.path} at line {self.line_number}'

    def __str__(self) -> str:
        ret = f'**OpenFileAction**\n'
        if self.path:
            ret += f'Path: [{self.path}]\n'
        if self.line_number:
            ret += f'Line: {self.line_number}\n'
        if self.context_lines:
            ret += f'Context lines: {self.context_lines}\n'
        if self.thought:
            ret += f'Thought: {self.thought}\n'
        return ret


@dataclass
class GotoLineAction(BaseAction):
    """Jumps to a specific line in the currently open file.
    
    Attributes:
        line_number (int): The line number to jump to (1-indexed).
        thought (str): The reasoning behind the goto action.
    """

    action_type: ActionType = ActionType.FILE_GOTO_LINE
    line_number: int = field(
        default=1,
        metadata={
            'tool_param': True,
            'description': 'The line number to jump to (1-indexed).',
            'required': True
        }
    )
    thought: str = ''
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None
    description: str = "Jump to a specific line in the currently open file and show the content around the line."

    @property
    def message(self) -> str:
        return f'Going to line: {self.line_number}'

    def __str__(self) -> str:
        ret = f'**GotoLineAction**\n'
        if self.line_number:
            ret += f'Line: {self.line_number}\n'
        if self.thought:
            ret += f'Thought: {self.thought}\n'
        return ret


@dataclass
class FileScrollDownAction(BaseAction):
    """Scrolls down in the currently open file by the window size (default 100 lines).
    
    Attributes:
        thought (str): The reasoning behind the scroll down action.
    """

    action_type: ActionType = ActionType.FILE_SCROLL_DOWN
    thought: str = ''
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None
    description: str = "Scroll down 100 lines in the currently open file."

    @property
    def message(self) -> str:
        return 'Scrolling down in the current file'

    def __str__(self) -> str:
        ret = f'**FileScrollDownAction**\n'
        if self.thought:
            ret += f'Thought: {self.thought}\n'
        return ret


@dataclass
class FileScrollUpAction(BaseAction):
    """Scrolls up in the currently open file by the window size (default 100 lines).
    
    Attributes:
        thought (str): The reasoning behind the scroll up action.
    """

    action_type: ActionType = ActionType.FILE_SCROLL_UP
    thought: str = ''
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None
    description: str = "Scroll up 100 lines in the currently open file"

    @property
    def message(self) -> str:
        return 'Scrolling up in the current file'

    def __str__(self) -> str:
        ret = f'**FileScrollUpAction**\n'
        if self.thought:
            ret += f'Thought: {self.thought}\n'
        return ret


@dataclass
class CreateFileAction(BaseAction):
    """Creates a new file with the specified content.
    
    Attributes:
        filename (str): The name/path of the file to create.
        content (str): The content to write to the new file. Default is empty string.
        thought (str): The reasoning behind the file creation.
    """

    action_type: ActionType = ActionType.CREATE_FILE
    filename: str = field(
        default="",
        metadata={
            'tool_param': True,
            'description': 'The name/path of the file to create.',
            'required': True
        }
    )
    content: str = field(
        default="",
        metadata={
            'tool_param': True,
            'description': 'The content to write to the new file.',
            'required': True
        }
    )
    thought: str = ''
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None
    description: str = "Create a new file with the specified content, it will also replace the original file if it already exists."

    @property
    def message(self) -> str:
        return f'Creating file: {self.filename}'

    def __str__(self) -> str:
        ret = f'**CreateFileAction**\n'
        if self.filename:
            ret += f'Filename: {self.filename}\n'
        if self.thought:
            ret += f'Thought: {self.thought}\n'
        if self.content:
            ret += f'Content:\n```\n{self.content}\n```\n'
        return ret


@dataclass
class SearchDirAction(BaseAction):
    """Searches for a text pattern in all files within a directory.
    
    Attributes:
        search_term (str): The text to search for.
        dir_path (str): The directory path to search in. Default is current directory.
        thought (str): The reasoning behind the search action.
    """

    action_type: ActionType = ActionType.SEARCH_DIR
    search_term: str = field(
        default="",
        metadata={
            'tool_param': True,
            'description': 'The text to search for.',
            'required': True
        }
    )
    dir_path: str = field(
        default="",
        metadata={
            'tool_param': True,
            'description': 'The directory path to search in.',
            'required': True
        }
    )
    thought: str = ''
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None
    description: str = "Search for a text pattern in all files within a directory"

    @property
    def message(self) -> str:
        return f'Searching for "{self.search_term}" in directory: {self.dir_path}'

    def __str__(self) -> str:
        ret = f'**SearchDirAction**\n'
        if self.search_term:
            ret += f'Search term: "{self.search_term}"\n'
        if self.dir_path:
            ret += f'Directory: {self.dir_path}\n'
        if self.thought:
            ret += f'Thought: {self.thought}\n'
        return ret


@dataclass
class SearchFileAction(BaseAction):
    """Searches for a text pattern in a specific file or the currently open file.
    
    Attributes:
        search_term (str): The text to search for.
        file_path (Optional[str]): The file path to search in. If None, searches in currently open file.
        thought (str): The reasoning behind the search action.
    """

    action_type: ActionType = ActionType.SEARCH_FILE
    search_term: str = field(
        default="",
        metadata={
            'tool_param': True,
            'description': 'The text to search for.',
            'required': True
        }
    )
    file_path: Optional[str] = field(
        default="",
        metadata={
            'tool_param': True,
            'description': 'The file path to search in. If None, searches in currently open file.',
            'required': True
        }
    )
    thought: str = ''
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None
    description: str = "Searches for a text pattern in a specific file or the currently open file."

    @property
    def message(self) -> str:
        if self.file_path:
            return f'Searching for "{self.search_term}" in file: {self.file_path}'
        else:
            return f'Searching for "{self.search_term}" in current file'

    def __str__(self) -> str:
        file_info = self.file_path if self.file_path else 'current file'
        ret = f'**SearchFileAction**\n'
        if self.search_term:
            ret += f'Search term: "{self.search_term}"\n'
        if file_info:
            ret += f'File: {file_info}\n'
        if self.thought:
            ret += f'Thought: {self.thought}\n'
        return ret


@dataclass
class FindFileAction(BaseAction):
    """Finds files by name pattern within a directory.
    
    Attributes:
        file_name (str): The file name or pattern to search for.
        dir_path (str): The directory path to search in. Default is current directory.
        thought (str): The reasoning behind the find action.
    """

    action_type: ActionType = ActionType.FIND_FILE
    file_name: str = field(
        default="",
        metadata={
            'tool_param': True,
            'description': 'The file name or pattern to search for.',
            'required': True
        }
    )
    dir_path: str = field(
        default="",
        metadata={
            'tool_param': True,
            'description': 'The directory path to search in.',
            'required': True
        }
    )
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None
    description: str = "Finds files by name pattern within a directory."

    @property
    def message(self) -> str:
        return f'Finding files named "{self.file_name}" in directory: {self.dir_path}'

    def __str__(self) -> str:
        ret = f'**FindFileAction**\n'
        if self.file_name:
            ret += f'File name: "{self.file_name}"\n'
        if self.dir_path:
            ret += f'Directory: {self.dir_path}\n'
        if self.thought:
            ret += f'Thought: {self.thought}\n'
        return ret


@dataclass
class ListFilesAction(BaseAction):
    """Lists all files and directories in a specified path.
    
    Attributes:
        path (str): The directory path to list contents of. Default is current directory.
        show_hidden (bool): Whether to show hidden files/directories. Default is False.
        thought (str): The reasoning behind the list action.
    """

    path: str = field(
        default=".",
        metadata={
            'tool_param': True,
            'description': 'The directory path to list contents of.',
            'required': True
        }
    )
    show_hidden: bool = field(
        default=False,
        metadata={
            'tool_param': True,
            'description': 'Whether to show hidden files/directories.',
            'required': True
        }
    )
    action_type: ActionType = ActionType.LIST_FILES
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None
    description: str = "List all files and directories in a specified path"

    @property
    def message(self) -> str:
        return f'Listing files in directory: {self.path}'

    def __str__(self) -> str:
        ret = f'**ListFilesAction**\n'
        if self.path:
            ret += f'Path: {self.path}\n'
        if self.show_hidden:
            ret += f'Show hidden: {self.show_hidden}\n'
        if self.thought:
            ret += f'Thought: {self.thought}\n'
        return ret


@dataclass
class GetFileInfoAction(BaseAction):
    """Gets information about the currently open file.
    
    Attributes:
        thought (str): The reasoning behind the info request.
    """

    action_type: ActionType = ActionType.GET_FILE_INFO
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None
    description: str = "Get information about the currently open file"

    @property
    def message(self) -> str:
        return 'Getting current file information'

    def __str__(self) -> str:
        ret = f'**GetFileInfoAction**\n'
        if self.thought:
            ret += f'Thought: {self.thought}\n'
        return ret
