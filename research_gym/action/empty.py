from dataclasses import dataclass, field
from typing import ClassVar, Optional, Dict, Any, List

from research_gym.action.action import BaseAction, ActionConfirmationStatus, ActionSecurityRisk, ToolCallArguments
from research_gym.schema.action import ActionType
from time import sleep

@dataclass
class NullAction(BaseAction):
    """An action that does nothing."""
    call_id: str = '00000000'

    action_type: str = ActionType.NULL
    description: str = "Do nothing"
    error_message: str = ""

    @property
    def message(self) -> str:
        return 'No action'
    
    def __str__(self) -> str:
        return 'No action'


@dataclass
class SleepAction(BaseAction):
    sleep_time: float = field(
        default=10.0,
        metadata={
            'tool_param': True,
            'description': 'The time to sleep in seconds. (max: 1800 seconds)',
            'required': True
        }
    )
    action_type: str = ActionType.SLEEP
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.LOW
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    description: str = "Sleep for a specified time."

    @property
    def message(self) -> str:
        return f'Sleeping for {self.sleep_time} seconds...'

    def __str__(self) -> str:
        ret = f'**SleepAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'Sleep time: {self.sleep_time} seconds'
        return ret