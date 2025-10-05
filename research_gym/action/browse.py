from dataclasses import dataclass, field
from typing import ClassVar, Optional, Dict, Any, List

from research_gym.action.action import BaseAction, ActionConfirmationStatus, ActionSecurityRisk, ToolCallArguments
from research_gym.schema.action import ActionType


@dataclass
class WebPageGotoAction(BaseAction):
    """Navigate to URL and display content"""
    
    # Tool parameters: use metadata marking
    url: str = field(
        default='',
        metadata={
            'tool_param': True,
            'description': 'The URL to navigate to.',
            'required': True
        }
    )
    line_number: int = field(
        default=1,
        metadata={
            'tool_param': True,
            'description': f'The line number to start viewing from. The environment will perform line_number to line_number + 100 lines of content.',
            'required': False
        }
    )
    
    action_type: str = ActionType.WEB_BROWSE
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.LOW
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    description: str = "Navigate to a webpage based on URL and display some of its content. The environment will cache the webpage content for another action to use until perform next web_browse action."

    @property
    def message(self) -> str:
        return f'Navigating to: {self.url}'

    def __str__(self) -> str:
        ret = f'**GotoAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'URL: {self.url}\n'
        ret += f'Line number: {self.line_number}'
        return ret


@dataclass
class WebPageGotoLineAction(BaseAction):
    """Jump to specified line"""
    
    line_number: int = field(
        default=1,
        metadata={
            'tool_param': True,
            'description': 'The line number to jump to.',
            'required': True
        }
    )
    
    action_type: str = ActionType.WEB_PAGE_GOTO_LINE
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.LOW
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    description: str = "Jump to a specific line in the current page"

    @property
    def message(self) -> str:
        return f'Jumping to line: {self.line_number}'

    def __str__(self) -> str:
        ret = f'**GotoLineAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'Line number: {self.line_number}'
        return ret


@dataclass
class WebPageScrollDownAction(BaseAction):
    """Scroll down page"""
    
    action_type: str = ActionType.WEB_PAGE_SCROLL_DOWN
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.LOW
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    description: str = "Scroll down the current page. It will scroll down 100 lines of content."

    @property
    def message(self) -> str:
        return 'Scrolling down the page'

    def __str__(self) -> str:
        ret = f'**ScrollDownAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += 'Action: Scroll down'
        return ret


@dataclass
class WebPageScrollUpAction(BaseAction):
    """Scroll up page"""
    
    action_type: str = ActionType.WEB_PAGE_SCROLL_UP
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.LOW
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    description: str = "Scroll up the current page. It will scroll up 100 lines of content."

    @property
    def message(self) -> str:
        return 'Scrolling up the page'

    def __str__(self) -> str:
        ret = f'**ScrollUpAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += 'Action: Scroll up'
        return ret


@dataclass
class WebPageSearchAction(BaseAction):
    """Search keywords in current page"""
    
    keyword: str = field(
        default='',
        metadata={
            'tool_param': True,
            'description': 'The keyword to search for in the current page.',
            'required': True
        }
    )
    context_lines: int = field(
        default=5,
        metadata={
            'tool_param': True,
            'description': 'The number of context lines to show around each match.',
            'required': False
        }
    )
    
    action_type: str = ActionType.WEB_PAGE_SEARCH
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.LOW
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    description: str = "Search for a keyword in the current page. It will search for the keyword in the current page and display the context lines around the place where the keyword appear for the first time."

    @property
    def message(self) -> str:
        return f'Searching for: {self.keyword}'

    def __str__(self) -> str:
        ret = f'**SearchInPageAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        if self.keyword:
            ret += f'Keyword: {self.keyword}\n'
        if self.context_lines:
            ret += f'Context lines: {self.context_lines}'
        return ret


@dataclass
class WebPageSearchNextAction(BaseAction):
    """Jump to next search result"""
    
    context_lines: int = field(
        default=5,
        metadata={
            'tool_param': True,
            'description': 'The number of context lines to show around the match.',
            'required': False
        }
    )
    search_index: Optional[int] = field(
        default=None,
        metadata={
            'tool_param': True,
            'description': 'The index of the search result to jump to.',
            'required': False
        }
    )
    
    action_type: str = ActionType.WEB_PAGE_SEARCH_NEXT
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.LOW
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    description: str = "Jump to the place where the keyword appares for the {search_index} time in the last opened web page. If search_index is bigger than the number of matches, it will jump to the (search_index % the number of matches) th match."

    @property
    def message(self) -> str:
        if self.search_index is not None:
            return f'Jumping to search result #{self.search_index}'
        return 'Jumping to next search result'

    def __str__(self) -> str:
        ret = f'**SearchNextAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        if self.context_lines:
            ret += f'Context lines: {self.context_lines}\n'
        if self.search_index is not None:
            ret += f'Search index: {self.search_index}'
        else:
            ret += 'Search index: next'
        return ret


@dataclass
class WebPageGetLinksAction(BaseAction):
    """Get page links"""
    
    page_size: int = field(
        default=10,
        metadata={
            'tool_param': True,
            'description': 'The number of links to show per page.',
            'required': False
        }
    )
    page_number: int = field(
        default=1,
        metadata={
            'tool_param': True,
            'description': 'The page number to display.',
            'required': False
        }
    )
    
    action_type: str = ActionType.WEB_PAGE_GET_LINKS
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.LOW
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    description: str = "Get hyperlinks from the current page"

    @property
    def message(self) -> str:
        return f'Getting links (page {self.page_number}, {self.page_size} per page)'

    def __str__(self) -> str:
        ret = f'**GetLinksAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        if self.page_size:
            ret += f'Page size: {self.page_size}\n'
        if self.page_number:
            ret += f'Page number: {self.page_number}'
        return ret

