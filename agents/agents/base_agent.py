"""
Agent base class definition
Provide Agent's common interface and basic function implementation
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import yaml

from agents.context.context_managers.base_manager import BaseManager
from agents.context.nodes import NodeType

# Import context related types
from agents.config.types import TaskStatus

# Import LLM related types
from llm.config import AgentConfig
from llm.llm_client import LLMClient

# Import environment related types
from research_gym.observation.observation import BaseObservation
from research_gym.action.action import BaseAction
from research_gym.schema import ActionType
from research_gym.configs.task_config import TaskConfig


class BaseAgent(ABC):
    """
    Agent base class, defines Agent's common interface and basic functions

    All concrete Agent implementations should inherit this base class and implement abstract methods
    """
    
    def __init__(
        self, 
        context_manager: BaseManager, 
        llm_client: LLMClient, 
        config: AgentConfig,
        task_config: TaskConfig,
    ):
        """
        Initialize Agent base class

        Args:
            context_manager: Context manager
            llm_client: LLM client
            config: Configuration object
        """
        # Basic configuration
        self.config = config
        self.task_config = task_config
        print("self.config.context_tools: ", self.config.context_tools)
        print("self.config.env_tools: ", self.config.env_tools)
        # Core components
        self.context_manager = context_manager
        self.llm_client = llm_client
        
        # Tool system
        self.context_management_actions: List[ActionType] = self._init_context_tool_names()
        self.environment_actions: List[ActionType] = self._init_env_tool_names()
        print("self.context_management_actions: ", self.context_management_actions)
        print("self.environment_actions: ", self.environment_actions)
        self.tools: List[BaseAction] = []
        
        self._initialize_tools()
        
        # Debug information
        self.last_llm_response: Optional[Dict[str, Any]] = None
        self.last_llm_messages: List[Dict[str, Any]] = []
    
    def _init_context_tool_names(self) -> List[ActionType]:
        tools = [tool for tool in self.config.context_tools]
        action_types = [ActionType[tool] for tool in tools if tool in ActionType.__members__]
        return action_types
    
    def _init_env_tool_names(self) -> List[ActionType]:
        tools = [tool for tool in self.config.env_tools]
        action_types = [ActionType[tool] for tool in tools if tool in ActionType.__members__]
        return action_types
    
    def _initialize_tools(self):
        """
        Initialize Action-based tool system

        Args:
            tool_names: Tool name list, use default tools when None
        """
        from research_gym.action import get_action_class
        self.tools = [get_action_class(tool_name) for tool_name in self.context_management_actions + self.environment_actions]
    
    @abstractmethod
    async def step(
        self, 
        observation: BaseObservation, 
        cli_message: str = "", 
        ui_message: str = ""
    ) -> Tuple[BaseAction, Dict[str, Any]]:
        """
        Execute one agent step (abstract method)

        Args:
            observation: Environment observation result
            cli_message: CLI message
            ui_message: UI message

        Returns:
            Tuple containing:
            - BaseAction: Action to execute
            - Dict[str, Any]: Information for user
        """
        pass

    @abstractmethod
    def _handle_internal_action(self, action: BaseAction) -> BaseAction:
        """
        Handle internal actions (abstract method)
        """
        pass
    
    def is_task_completed(self) -> bool:
        """
        Check if task is completed

        Returns:
            True if task is completed, False otherwise
        """
        return self.context_manager.current_node.node_type == NodeType.DONE
    
    def get_step_info(self) -> Dict[str, Any]:
        """
        Get current step information

        Returns:
            Dictionary containing step information
        """
        return {
            "provider": self.config.default_provider,
            "completed": self.is_task_completed()
        }
    
    def get_information_for_user(self) -> Dict[str, Any]:
        """
        Get information for user
        Integrate context_manager information and agent status information

        Returns:
            Information dictionary for user
        """
        information = self.context_manager.get_information_for_user()
        information.update(self.get_step_info())
        return information
    
    def update_debug_info(self, response: Dict[str, Any], messages: List[Dict[str, Any]]):
        """
        Update debug information

        Args:
            response: LLM response
            messages: LLM message list
        """
        self.last_llm_response = response
        self.last_llm_messages = messages
    
    def clear_debug_info(self):
        """Clear debug information"""
        self.last_llm_response = None
        self.last_llm_messages = []
    
    def reset(self):
        """
        Reset agent state
        """
        self.clear_debug_info()
        # Note: Do not reset context_manager as it may contain important context information
    
    def get_agent_state(self) -> Dict[str, Any]:
        """
        Get agent's current state

        Returns:
            Dictionary containing agent state
        """
        return {
            "is_completed": self.is_task_completed(),
            "tools_count": len(self.tools),
        }

    def _to_dict(self) -> Dict[str, Any]:
        """
        Convert agent to dictionary
        """
        return {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert agent to dictionary
        """
        return self._to_dict()
    
    def from_dict(self, d: Dict[str, Any]):
        """
        Restore agent from dictionary
        """
        pass