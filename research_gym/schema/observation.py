from enum import Enum


class ObservationType(str, Enum):
    BASE = "base"
    
    COMMAND = "command"
    # Detailed command observation types
    COMMAND_CREATE_SESSION = "command_create_session"
    COMMAND_LIST_SESSIONS = "command_list_sessions"
    COMMAND_RUN = "command_run"
    COMMAND_INPUT_IN_SESSION = "command_input_in_session"
    COMMAND_OUTPUT_SESSION = "command_output_session"
    COMMAND_SESSION_STATUS = "command_session_status"
    COMMAND_CLEAR_SESSION_BUFFER = "command_clear_session_buffer"
    COMMAND_CLOSE_SESSION = "command_close_session"
    COMMAND_KILL_PROCESSES = "command_kill_processes"
    
    PARSE = "parse"

    BROWSE = 'browse'
    """The HTML content of a URL
    """

    RUN = 'run'
    """The output of a command
    """

    CHAT = 'chat'
    """A message from the user
    """

    DELEGATE = 'delegate'
    """The result of a task delegated to another agent
    """

    MESSAGE = 'message'

    FILE = "file"

    FILE_READ = 'file_read'
    """The content of a file
    """

    FILE_WRITE = 'file_write'

    FILE_ERROR = 'file_error'
    """The result of a error operation"""

    FILE_INFO = 'file_info'
    """Result of a file info operation"""

    FILE_LIST = 'file_list'
    """Result of a file list operation"""

    FILE_SEARCH = 'file_search'
    """Result of a file search operation"""

    NULL = 'null'

    USER_REJECTED = 'user_rejected'

    CONDENSE = 'condense'
    """Result of a condensation operation."""

    RECALL = 'recall'
    """Result of a recall operation. This can be the workspace context, a microagent, or other types of information."""

    MCP = 'mcp'
    """Result of a MCP Server operation"""

    WEB_SEARCH = 'web_search'
    """Result of a web search operation"""

    WEB_BROWSE = 'web_browse'
    """Result of a web browsing operation"""
    # Detailed web browsing observation types
    WEB_BROWSE_READ = 'web_browse_read'
    WEB_BROWSE_SEARCH = 'web_browse_search'
    WEB_BROWSE_LINKS = 'web_browse_links'

    EVAL = "eval"
    """Result of a evaluation operation"""
