from enum import Enum


class ActionType(str, Enum):
    
    # system
    SUMMARIZE = 'summarize'
    """Summarizes the current state.
    """
    
    INTERNAL_SUMMARIZE = 'internal_summarize'
    """Summarizes the current state.
    """

    THINK = 'think'
    """Logs a thought.
    """

    # command
    RUN = 'run_command'
    """Runs a command.
    """
    
    EDIT = 'edit_file'
    """Edits a file by providing a draft.
    """

    CREATE_SESSION = 'create_session'
    """Creates a new session.
    """

    LIST_SESSIONS = 'list_sessions'
    """Lists all sessions.
    """

    INPUT_IN_SESSION = 'input_in_session'
    """Inputs a message in a session.
    """

    GET_SESSION_STATUS = 'get_session_status'
    """Gets the status of a session.
    """

    GET_SESSION_OUTPUT = 'get_session_output'
    """Gets the output of a session.
    """

    GET_SESSION_RECENT_OUTPUT = 'get_session_recent_output'
    """Gets the recent output of a session.
    """

    CHECK_SESSION_IDLE = 'check_session_idle'
    """Checks if a session is idle.
    """

    CLEAR_SESSION_BUFFER = 'clear_session_buffer'
    """Clears the buffer of a session.
    """

    CLOSE_SESSION = 'close_session'
    """Closes a session.
    """

    CLOSE_ALL_SESSIONS = 'close_all_sessions'
    """Closes all sessions.
    """
    
    KILL_SESSION_PROCESSES = 'kill_session_processes'
    """Kills all processes of a session.
    """

    # file
    OPEN_FILE = 'open_file'
    """Opens a file.
    """

    FILE_GOTO_LINE = 'file_goto_line'
    """Goes to a line in a file.
    """

    FILE_SCROLL_DOWN = 'file_scroll_down'
    """Scrolls down in a file.
    """

    FILE_SCROLL_UP = 'file_scroll_up'
    """Scrolls up in a file.
    """

    CREATE_FILE = 'create_file'
    """Creates a new file.
    """

    SEARCH_DIR = 'search_dir'
    """Searches for a directory.
    """

    SEARCH_FILE = 'search_file'
    """Searches for a file.
    """

    FIND_FILE = 'find_file'
    """Finds a file.
    """

    LIST_FILES = 'list_files'
    """Lists all files in a directory.
    """

    GET_FILE_INFO = 'get_file_info'
    """Gets information about a file.
    """
    
    # web search
    WEB_SEARCH = 'web_search'
    """Searches the web for information."""
    
    # web browse
    WEB_BROWSE = 'web_browse'   
    """Opens a web page."""

    WEB_PAGE_GOTO_LINE = 'web_page_goto_line'
    """Goes to a line in the last opened web page."""
    
    WEB_PAGE_SCROLL_DOWN = 'web_page_scroll_down'
    """Scrolls down in the last opened web page."""
    
    WEB_PAGE_SCROLL_UP = 'web_page_scroll_up'
    """Scrolls up in the last opened web page."""
    
    WEB_PAGE_SEARCH = 'web_page_search'
    """Searches for a keyword in the last opened web page."""
    
    WEB_PAGE_SEARCH_NEXT = 'web_page_search_next'
    """Searches for the place where the keyword appares for the {search_index} time in the last opened web page."""
    
    WEB_PAGE_GET_LINKS = 'web_page_get_links'
    """Gets the links in the last opened web page."""
    
    SLEEP = 'sleep'
    """Sleeps for a specified time."""
    
    FINISH = 'finish'
    """If you're absolutely certain that you've completed your task and have tested your work,
    use the finish action to stop working.
    """

    REJECT = 'reject'
    """If you're absolutely certain that you cannot complete the task with given requirements,
    use the reject action to stop working.
    """

    NULL = 'null'

    # PAUSE = 'pause'
    """Pauses the task.
    """

    # RESUME = 'resume'
    """Resumes the task.
    """

    # STOP = 'stop'
    """Stops the task. Must send a start action to restart a new task.
    """

    # CHANGE_AGENT_STATE = 'change_agent_state'

    # PUSH = 'push'
    """Push a branch to github."""

    # SEND_PR = 'send_pr'
    """Send a PR to github."""

    # RECALL = 'recall'
    """Retrieves content from a user workspace, microagent, or other source."""

    # CONDENSATION = 'condensation'
    """Condenses a list of events into a summary."""

    PARSE_PDF = 'parse_pdf'
    """Parses a PDF file."""

    PARSE_DOCX = 'parse_docx'
    """Parses a DOCX file."""

    PARSE_LATEX = 'parse_latex'
    """Parses a LaTeX file."""

    PARSE_AUDIO = 'parse_audio'
    """Parses an audio file."""

    PARSE_IMAGE = 'parse_image'
    """Parses an image file."""

    PARSE_VIDEO = 'parse_video'
    """Parses a video file."""

    PARSE_PPTX = 'parse_pptx'
    """Parses a PPTX file."""

    EVAL = 'eval'
    """Evaluates the agent's output.
    """

    VIEW_HINT = 'view_hint'
    """Views the hint."""