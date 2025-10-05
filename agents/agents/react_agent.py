"""
Ranger Agent implementation
"""

import json
import re
import yaml
from typing import List, Dict, Any, Optional, Tuple

from agents.agents.base_agent import BaseAgent
from agents.context.nodes.base_node import NodeType
from agents.config.types import TaskStatus

from research_gym.observation.observation import BaseObservation
from research_gym.action.action import BaseAction, ToolCall
from agents.utils.utils import make_toolcall_id
from research_gym.schema import ActionType
from research_gym.action import get_action_class
from research_gym.action.system import FinishAction, AgentFinishTaskCompleted, SummarizeAction
from research_gym.configs.task_config import TaskConfig
from research_gym.action.empty import NullAction

from agents.context.context_managers import BaseManager
from agents.context.nodes.base_node import NodeType

from agents.config.types import TaskStatus
from llm.llm_basics import LLMUsage, LLMMessage, LLMResponse

from llm.config import AgentConfig
from llm.llm_client import LLMClient


class ReActAgent(BaseAgent):
    """
    ReAct Agent implementation

    Based on Reasoning and Acting mode, alternately execute reasoning (Thought) and action (Action)
    """
    
    def __init__(self, 
        context_manager: BaseManager, 
        llm_client: LLMClient, 
        config: AgentConfig,
        task_config: TaskConfig,
    ):
        """Initialize ReAct Agent"""
        super().__init__(
            context_manager,
            llm_client,
            config,
            task_config)
        
        # ReAct specific states
        self.internal_action_types = [ActionType.SUMMARIZE, ActionType.THINK]
        self.internal_node_types = [NodeType.SUMMARY]
        self.internal_action_count = 0 # Record the number of internal actions, if it exceeds a certain number, it is considered to be failed
        self.think_cooldown_active = False
        print("self.tools: ", self.tools)
        

    async def step(self, observation: BaseObservation = None, cli_message: str = "", ui_message: str = "", final_step: bool = False) -> Tuple[BaseAction, Dict[str, Any]]:
        """
        Execute one agent step

        Args:
            observation: Environment observation result
            cli_message: CLI message
            ui_message: UI message

        Returns:
            actions: Parsed action list
            information_for_user: Information for user
        """
        try:
            # 1. Record step start
            # Current node must be a react node
            print(f"node type: {self.context_manager.current_node.node_type}")
            if final_step:
                self.context_manager.add_observation(observation, cli_message, ui_message)
                return None, None
            # 2. Check if already completed (before calling LLM)
            if self.is_task_completed():
                # Task completed, no need to call LLM, return completion status directly
                self.clear_debug_info()
                
                information_for_user = self.get_information_for_user()
                information_for_user["completed"] = True
                
                return FinishAction(thought="Task completed", task_completed=AgentFinishTaskCompleted.TRUE), information_for_user

            # 3. Update observation to context manager
            # print("observation: ", observation)
            self.context_manager.add_observation(observation, cli_message, ui_message)
            
            # 4. Current node must be react, next must also be react node, so should update a react node
            self.context_manager.update_status(NodeType.REACT)
            
            if self.context_manager.should_summarize():
                random_tool_call_id = make_toolcall_id()
                tool_calls = [ToolCall(call_id=random_tool_call_id, name=ActionType.SUMMARIZE.value, arguments={"start_summary_depth": 1, "end_summary_depth": (self.context_manager.current_node.depth+1)//2})]
                response = LLMResponse(role="assistant", content="", tool_calls=tool_calls)
            else:

                # 5. Get messages to send to LLM, if too long there will only be summary action
                messages = self.context_manager.get_messages()

                # 6. Get tools that LLM can use
                tools = self.context_manager.get_tools_for_react(self.tools)

                # print("\nmessages: ", messages)

                # 6. Call LLM
                response = self.llm_client.chat(
                    messages=messages,
                    tools=tools,
                    reuse_history=False
                )

            # 7. Update response to context manager
            action = self.context_manager.add_response(response)

            if action.action_type in self.internal_action_types:
                action = self._handle_internal_action(action)
            
            # 8. Check if task completed (based on context_manager status)
            is_completed = self.is_task_completed()
            
            # 9. If task completed, return immediately, do not execute tools
            if is_completed:
                information_for_user = self.get_information_for_user()
                information_for_user["completed"] = True
                
                return FinishAction(thought="Task completed", task_completed=AgentFinishTaskCompleted.TRUE), information_for_user
            
            # 10. Get user information (mainly left side information)
            information_for_user = self.get_information_for_user()
            information_for_user["completed"] = False
            
            if self.think_cooldown_active:
                self.context_manager.think_cooldown = False

            return action, information_for_user
            
        except Exception as e:
            raise e
            error_msg = f"Error in agent step: {str(e)}"
            # Record error to context manager
            self.context_manager.current_step_info["error"] = error_msg
            # self.context_manager.record_agent_step()
            
            return NullAction(), {"error": error_msg}

    def _handle_internal_action(self, action: BaseAction) -> BaseAction:
        
        fail_count = 0
        while action.action_type in self.internal_action_types or self.context_manager.current_node.node_type in self.internal_node_types:
            if self.context_manager.current_node.node_type in self.internal_node_types:
                success = self.context_manager.add_response(response)
                if success:
                    if self.context_manager.current_node.node_type == NodeType.SUMMARY:
                        # Find previous sibling
                        current_node = self.context_manager.current_node.parent.children[-2]
                        
                        # Recursively copy from where it hasn't been summarized
                        while current_node.depth < action.end_summary_depth:
                            if len(current_node.children) > 0:
                                current_node = current_node.children[-1]
                            else:
                                current_node = None
                                break
                        
                        if current_node is not None:
                            if current_node.node_type == NodeType.REACT:
                                current_node.duplicate(parent=self.context_manager.current_node)
                            while len(self.context_manager.current_node.children) > 0:
                                self.context_manager.current_node = self.context_manager.current_node.children[-1]
                        self.context_manager.update_status(NodeType.REACT)
                        messages = self.context_manager.get_messages()
                        tools = self.context_manager.get_tools_for_react(self.tools)
                else:
                    fail_count += 1
                    if fail_count > 10:
                        raise ValueError(f"Invalid internal action call!")
            else:
                if action.action_type == ActionType.SUMMARIZE:
                    validation_result = self.context_manager.check_internal_actions_validation(action_type=ActionType.SUMMARIZE, depth1=action.start_summary_depth, depth2=action.end_summary_depth)
                    if validation_result["is_valid"]:
                        self.internal_action_count += 1
                        self.context_manager.update_status(NodeType.SUMMARY, parent_node=validation_result["parent_node"])
                        messages = self.context_manager.get_messages(depth1=action.start_summary_depth, depth2=action.end_summary_depth)
                        tools = self.context_manager.get_tools_for_internal_action()
                    else:
                        raise ValueError(f"The last action is not valid, reason: \n {validation_result['reason']}")
                elif action.action_type == ActionType.THINK:
                    validation_result = self.context_manager.check_internal_actions_validation(action_type=ActionType.THINK)
                    if validation_result["is_valid"]:
                        # Set next round cooldown, add user prompt message, continue next round REACT
                        self.internal_action_count += 1
                        self.context_manager.think_cooldown = True
                        obs = BaseObservation.from_result(
                            result={"success": True, "message": "OK, your thought has been logged. Go ahead. Do not think more until you generate tools and interact with the real environment"},
                            action=action
                        )
                        # Add user prompt message
                        self.context_manager.add_observation(obs)
                        # Enable next round REACT
                        self.context_manager.update_status(NodeType.REACT)
                        messages = self.context_manager.get_messages()
                        tools = self.context_manager.get_tools_for_react(self.tools)
                        # Consume current cooldown: this round is disabled, and the tool is restored after taking it
                        self.think_cooldown_active = getattr(self.context_manager, 'think_cooldown', False)
                    else:
                        obs = BaseObservation.from_result(result={"success": False, "message": f"The last action is not valid, reason: \n {validation_result['reason']}"}, action=action)
                        self.context_manager.add_observation(obs)
                        self.context_manager.update_status(NodeType.REACT)
                        messages = self.context_manager.get_messages()
                        tools = self.context_manager.get_tools_for_react(self.tools)
                        
                else:
                    raise ValueError(f"Invalid action type: {action.action_type}")
                
                if self.context_manager.limits.max_internal_action_times > 0 and self.internal_action_count > self.context_manager.limits.max_internal_action_times:
                    self.context_manager.update_status(NodeType.DONE)
                    return FinishAction(thought="Internal action count exceeded", task_completed=AgentFinishTaskCompleted.FALSE)


            if self.context_manager.current_node.node_type == NodeType.REACT:
                response = self.llm_client.chat(
                    messages=messages,
                    tools=tools,
                    reuse_history=False
                )
                action = self.context_manager.add_response(response)
                print("_handle_internal_action -> action: ", action)
                if action.action_type in self.internal_action_types:
                    continue
                else:
                    return action
            else:
                response = self.llm_client.chat(
                    messages=messages,
                    tools=tools,
                    reuse_history=False
                )

        return action

    def reset(self):
        """Reset Ranger Agent state"""
        super().reset()

    def _to_dict(self) -> Dict[str, Any]:
        """
        Convert agent to dictionary
        """
        d = super()._to_dict()
        d["internal_action_count"] = self.internal_action_count
        return d
    
    def from_dict(self, d: Dict[str, Any]):
        """
        Restore agent from dictionary
        """
        super().from_dict(d)
        self.internal_action_count = d["internal_action_count"]