"""
ActionType to Action Class Mapping
Provide mapping relationship from ActionType enum values to specific Action classes
"""

from typing import Dict, Type, Optional
from research_gym.schema.action import ActionType
from research_gym.action.action import BaseAction

# Import all Action classes
from research_gym.action.files import (
    FileEditAction,
    OpenFileAction,
    GotoLineAction,
    FileScrollDownAction,
    FileScrollUpAction,
    CreateFileAction,
    SearchDirAction,
    SearchFileAction,
    FindFileAction,
    ListFilesAction,
    GetFileInfoAction,
)

from research_gym.action.commands import (
    RunCommandAction,
    CreateSessionAction,
    ListSessionsAction,
    InputInSessionAction,
    GetSessionOutputAction,
    GetSessionRecentOutputAction,
    SessionStatusAction,
    SessionIdleAction,
    ClearSessionBufferAction,
    CloseSessionAction,
    CloseAllSessionsAction,
    KillSessionProcessesAction,
)

from research_gym.action.parses import (
    ParsePdfAction,
    ParseDocxAction,
    ParseLatexAction,
    ParseAudioAction,
    ParseImageAction,
    ParseVideoAction,
    ParsePptxAction,
)

from research_gym.action.system import (
    FinishAction,
    SummarizeAction,
    EvalAction,
    ViewHintAction,
    ThinkAction,
)

from research_gym.action.empty import NullAction, SleepAction

from research_gym.action.search import SearchAction

from research_gym.action.browse import (
    WebPageGotoAction, WebPageGotoLineAction, WebPageScrollDownAction,
    WebPageScrollUpAction, WebPageSearchAction, WebPageSearchNextAction,
    WebPageGetLinksAction
)

# ActionType to Action class mapping table
ACTION_TYPE_TO_CLASS: Dict[ActionType, Type[BaseAction]] = {
    # File operations
    ActionType.EDIT: FileEditAction,
    ActionType.CREATE_FILE: CreateFileAction,
    ActionType.FILE_GOTO_LINE: GotoLineAction,
    ActionType.FILE_SCROLL_DOWN: FileScrollDownAction,
    ActionType.FILE_SCROLL_UP: FileScrollUpAction,
    ActionType.SEARCH_DIR: SearchDirAction,
    ActionType.SEARCH_FILE: SearchFileAction,
    ActionType.FIND_FILE: FindFileAction,
    ActionType.LIST_FILES: ListFilesAction,
    ActionType.GET_FILE_INFO: GetFileInfoAction,
    ActionType.OPEN_FILE: OpenFileAction,
    
    # Command execution
    ActionType.RUN: RunCommandAction,
    ActionType.CREATE_SESSION: CreateSessionAction,
    ActionType.LIST_SESSIONS: ListSessionsAction,
    ActionType.INPUT_IN_SESSION: InputInSessionAction,
    ActionType.GET_SESSION_STATUS: SessionStatusAction,
    ActionType.CHECK_SESSION_IDLE: SessionIdleAction,
    ActionType.CLEAR_SESSION_BUFFER: ClearSessionBufferAction,
    ActionType.CLOSE_SESSION: CloseSessionAction,
    ActionType.CLOSE_ALL_SESSIONS: CloseAllSessionsAction,
    ActionType.KILL_SESSION_PROCESSES: KillSessionProcessesAction,
    ActionType.GET_SESSION_OUTPUT: GetSessionOutputAction,
    ActionType.GET_SESSION_RECENT_OUTPUT: GetSessionRecentOutputAction,

    # Parsing operations
    ActionType.PARSE_PDF: ParsePdfAction,
    ActionType.PARSE_DOCX: ParseDocxAction,
    ActionType.PARSE_LATEX: ParseLatexAction,
    ActionType.PARSE_AUDIO: ParseAudioAction,
    ActionType.PARSE_IMAGE: ParseImageAction,
    ActionType.PARSE_VIDEO: ParseVideoAction,
    ActionType.PARSE_PPTX: ParsePptxAction,
    
    # System operations
    ActionType.FINISH: FinishAction,
    ActionType.THINK: ThinkAction,
    ActionType.NULL: NullAction,
    ActionType.SLEEP: SleepAction,
    ActionType.EVAL: EvalAction,
    
    # Search operations
    ActionType.WEB_SEARCH: SearchAction,
    
    # Web browsing operations
    ActionType.WEB_BROWSE: WebPageGotoAction,
    ActionType.WEB_PAGE_GOTO_LINE: WebPageGotoLineAction,
    ActionType.WEB_PAGE_SCROLL_DOWN: WebPageScrollDownAction,
    ActionType.WEB_PAGE_SCROLL_UP: WebPageScrollUpAction,
    ActionType.WEB_PAGE_SEARCH: WebPageSearchAction,
    ActionType.WEB_PAGE_SEARCH_NEXT: WebPageSearchNextAction,
    ActionType.WEB_PAGE_GET_LINKS: WebPageGetLinksAction,
    
    # Agent internal operations
    ActionType.SUMMARIZE: SummarizeAction,
    ActionType.VIEW_HINT: ViewHintAction,
}

# Reverse mapping: Action class to ActionType
CLASS_TO_ACTION_TYPE: Dict[Type[BaseAction], ActionType] = {
    action_class: action_type for action_type, action_class in ACTION_TYPE_TO_CLASS.items()
}


def get_action_class(action_type: ActionType) -> Optional[Type[BaseAction]]:
    """
    Get the corresponding Action class based on ActionType

    Args:
        action_type: ActionType enum value

    Returns:
        The corresponding Action class, or None if it doesn't exist
    """
    return ACTION_TYPE_TO_CLASS.get(action_type)


def get_action_type(action_class: Type[BaseAction]) -> Optional[ActionType]:
    """
    Get the corresponding ActionType based on Action class

    Args:
        action_class: Action class

    Returns:
        The corresponding ActionType enum value, or None if it doesn't exist
    """
    return CLASS_TO_ACTION_TYPE.get(action_class)


def create_action_by_type(action_type: ActionType, **kwargs) -> Optional[BaseAction]:
    """
    Create Action instance based on ActionType

    Args:
        action_type: ActionType enum value
        **kwargs: Parameters to pass to the Action constructor

    Returns:
        The created Action instance, or None if ActionType doesn't exist
    """
    action_class = get_action_class(action_type)
    if action_class:
        return action_class(**kwargs)
    return None


def get_supported_action_types() -> list[ActionType]:
    """
    Get all supported ActionType lists

    Returns:
        List of supported ActionType
    """
    return list(ACTION_TYPE_TO_CLASS.keys())


def get_action_info(action_type: ActionType) -> Optional[Dict[str, str]]:
    """
    Get basic information about Action class

    Args:
        action_type: ActionType enum value

    Returns:
        Dictionary containing Action class information, or None if it doesn't exist
    """
    action_class = get_action_class(action_type)
    if action_class:
        return {
            'action_type': action_type.value,
            'class_name': action_class.__name__,
            'module': action_class.__module__,
            'description': getattr(action_class, 'description', ''),
            'runnable': getattr(action_class, 'runnable', False),
        }
    return None


def print_mapping_table():
    """
    Print the complete mapping table for debugging
    """
    print("ActionType to Action Class Mapping:")
    print("=" * 50)
    for action_type, action_class in sorted(ACTION_TYPE_TO_CLASS.items(), key=lambda x: x[0].value):
        print(f"{action_type.value:<25} -> {action_class.__name__}")
    print("=" * 50)
    print(f"Total mappings: {len(ACTION_TYPE_TO_CLASS)}")


if __name__ == "__main__":
    # Test code
    print_mapping_table()
    
    # Test getting Action classes
    print("\nTest getting Action classes:")
    test_types = [ActionType.READ, ActionType.RUN, ActionType.THINK, ActionType.FINISH]
    for action_type in test_types:
        action_class = get_action_class(action_type)
        print(f"{action_type.value} -> {action_class.__name__ if action_class else 'Not Found'}")
    
    # Test creating Action instances
    print("\nTest creating Action instances:")
    read_action = create_action_by_type(ActionType.READ, path="test.py")
    run_action = create_action_by_type(ActionType.RUN, command="ls -la")
    print(f"ReadAction: {read_action}")
    print(f"RunAction: {run_action}") 