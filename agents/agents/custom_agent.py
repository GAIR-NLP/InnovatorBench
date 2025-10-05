"""
Base agent usage example
Demonstrate how to inherit BaseAgent base class to create custom Agent
"""

from typing import Tuple, Dict, Any
from agents.agents.base_agent import BaseAgent
from agents.context import ContextManager
from llm.llm_client import LLMClient
from llm.config import AgentConfig
from research_gym.observation.observation import BaseObservation
from research_gym.action.action import BaseAction
from research_gym.action.empty import NullAction


class CustomAgent(BaseAgent):
    """
    Custom Agent example, inherits BaseAgent base class
    """
    
    def get_default_system_prompt(self) -> str:
        """
        Implement abstract method: provide custom system prompt
        """
        return """You are a custom AI agent designed for specific tasks.
You have access to various tools and should use them wisely to accomplish your goals.

Always think step by step and explain your reasoning clearly."""
    
    async def step(
        self, 
        observation: BaseObservation, 
        cli_message: str = "", 
        ui_message: str = ""
    ) -> Tuple[BaseAction, Dict[str, Any]]:
        """
        Implement abstract method: execute one agent step
        """
        try:
            # 1. Increase step count
            self.step_count += 1

            # 2. Check if already completed
            if self.is_task_completed():
                self.context_manager.record_agent_step()
                self.clear_debug_info()
                
                information_for_user = self.get_information_for_user()
                information_for_user["completed"] = True
                
                from research_gym.action.system import FinishAction, AgentFinishTaskCompleted
                return FinishAction(
                    thought="Task completed successfully", 
                    task_completed=AgentFinishTaskCompleted.TRUE
                ), information_for_user
            
            # 3. Update observation to context manager
            self.context_manager.add_observation(observation, cli_message, ui_message)
            
            # 4. Get LLM messages
            messages = self.context_manager.get_messages()

            # 5. Call LLM
            response = self.llm_client.chat(
                messages=messages,
                tools=self.tools,
                reuse_history=False
            )
            
            # 6. Update debug information
            self.update_debug_info(response, messages)

            # 7. Process response and get action
            action = self.context_manager.add_response(response)

            # 8. Check if completed
            if self.is_task_completed():
                self.context_manager.record_agent_step()
                information_for_user = self.get_information_for_user()
                information_for_user["completed"] = True
                
                from research_gym.action.system import FinishAction, AgentFinishTaskCompleted
                return FinishAction(
                    thought="Task completed", 
                    task_completed=AgentFinishTaskCompleted.TRUE
                ), information_for_user
            
            # 9. Record step
            self.context_manager.record_agent_step()

            # 10. Get user information
            information_for_user = self.get_information_for_user()
            information_for_user["completed"] = False
            
            return action, information_for_user
            
        except Exception as e:
            error_msg = f"Error in custom agent step: {str(e)}"
            self.context_manager.current_step_info["error"] = error_msg
            self.context_manager.record_agent_step()
            
            return NullAction(), {"error": error_msg}
