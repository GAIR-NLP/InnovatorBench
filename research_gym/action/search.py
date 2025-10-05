from dataclasses import dataclass, field
from typing import ClassVar, Optional, Dict, Any, List

from research_gym.action.action import BaseAction, ActionConfirmationStatus, ActionSecurityRisk, ToolCallArguments
from research_gym.schema.action import ActionType


@dataclass
class SearchAction(BaseAction):
    """Web search action"""
    
    # Tool parameters: use metadata marking
    query: str = field(
        default='',
        metadata={
            'tool_param': True,
            'description': 'The search query to look up on the web.',
            'required': True
        }
    )
    top_k: int = field(
        default=10,
        metadata={
            'tool_param': True,
            'description': 'The maximum number of search results to return. If the number is bigger then 100, it will be set to 100.',
            'required': False
        }
    )
    
    action_type: str = ActionType.WEB_SEARCH
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.LOW
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    description: str = "Search the web for information in google/bing search engine"

    @property
    def message(self) -> str:
        return f'Searching for: {self.query[:100]}{"..." if len(self.query) > 100 else ""}'

    def __str__(self) -> str:
        ret = f'**SearchAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'Query: {self.query}\n'
        ret += f'Top K: {self.top_k}'
        return ret
