"""
ReAct context manager - Specialized for handling ReAct mode think-act-observe loops
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import re
from datetime import datetime

from agents.context.context_managers.base_manager import BaseManager
from agents.context.nodes.base_node import BaseNode, NodeType
from agents.context.nodes.react_node import ReActNode
from agents.context.nodes.summary_node import SummaryNode
from agents.config.types import TaskStatus
from research_gym.configs.task_config import TaskConfig
from llm.config import AgentConfig
from agents.context.prompt_builder import PromptBuilder
from agents.utils.utils import make_toolcall_id

from llm.llm_basics import LLMMessage, LLMResponse

from research_gym.configs.task_config import TaskConfig
from research_gym.action import BaseAction, get_action_class, ToolCall, ToolResult
from research_gym.schema import ActionType
from research_gym.observation.observation import BaseObservation
from research_gym.action.system import InternalSummarizeAction


class ReActManager(BaseManager):
    """
    ReAct context manager

    Specialized for managing ReAct mode think-act-observe loops,
    maintaining thought history, action history and current thought state.
    """
    
    def __init__(self, agent_config: AgentConfig, task_config: TaskConfig, node_types: List[NodeType] = None):
        """
        Initialize ReAct manager

        Args:
            agent_config: Agent configuration
            task_config: Task configuration
            node_types: Supported node types
        """
        
        super().__init__(agent_config, task_config, node_types)

        # self.prompt_builder = PromptBuilder(task_config)
        PromptBuilder.set_task_config(task_config)
        self.react_prompt_construct_mode = "react_default"
        # ThinkAction cooldown flag: if used in the previous round, disable it in this round
        self.think_cooldown: bool = False
        
    def _to_dict(self) -> Dict[str, Any]:
        """
        Convert context manager to dictionary
        """
        d = super()._to_dict()
        d["react_prompt_construct_mode"] = self.react_prompt_construct_mode
        d["think_cooldown"] = self.think_cooldown
        return d
    
    def _get_default_node_type(self) -> List[NodeType]:
        """
        Get default node types for ReAct mode
        """
        return [
            NodeType.ROOT,
            NodeType.REACT,
            NodeType.SUMMARY,
            NodeType.DONE
        ]
    
    def _initialize_context(self):
        """
        Initialize ReAct context structure
        """
        # Create root node
        self.root = ReActNode()
        self.root.node_type = NodeType.ROOT
        self.current_node = self.root
        self.total_nodes = 1
        self.logger.info("ReAct context manager initialization completed")

    def get_tools_for_react(self, tools: List[BaseAction]) -> List[BaseAction]:
        """
        Get tool list
        """
        print(f"self.react_prompt_construct_mode: {self.react_prompt_construct_mode}")
        
        # ThinkAction cooldown: if used in the last round, filter out ThinkAction this round
        if self.think_cooldown:
            tools = [tool for tool in tools if tool.action_type != ActionType.THINK]
        
        return tools

    def get_tools_for_internal_action(self) -> List[BaseAction]:
        print(f"get_tools_for_internal_action -> self.current_node.node_type: {self.current_node.node_type}")
        if self.current_node.node_type == NodeType.SUMMARY:
            return [InternalSummarizeAction]
        else:
            raise ValueError(f"Unsupported node type for internal action: {self.current_node.node_type}")

    def get_messages(self, depth1: int = None, depth2: int = None) -> List[LLMMessage]:
        """
        Build message list to send to LLM, dispatch to corresponding processing method according to current node type

        Returns:
            Formatted message list
        """
        # Dispatch to corresponding processing method according to current node type
        # TODO: update node messages after get_message
        if self.current_node.node_type == NodeType.ROOT:
            return self._get_react_messages()
        elif self.current_node.node_type == NodeType.REACT:
            return self._get_react_messages()
        elif self.current_node.node_type == NodeType.SUMMARY:
            return self._get_summary_messages(depth1, depth2)
        elif self.current_node.node_type == NodeType.DONE:
            return self._get_done_messages()
        else:
            raise ValueError(f"Unsupported node type: {self.current_node.node_type}")
    
    def _get_root_messages(self) -> List[LLMMessage]:
        """Get message list for ROOT node"""
        pass
    
    def _get_react_messages(self) -> List[LLMMessage]:
        """Get message list for REACT node"""
        messages = []
        path = self.current_node.get_path_from_root()

        # In react agent, check if context is too long, then directly assign SummarizeAction to action
        self.react_prompt_construct_mode = "react_default"

        system_prompt = PromptBuilder.build_system_prompt(
            self.current_node.node_type,
            prompt_construct_mode=self.react_prompt_construct_mode
        )
        messages.append(LLMMessage(role="system", content=system_prompt))

        remaining_seconds = self.task_config.max_working_time - (datetime.now() - self.start_time).total_seconds()
        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)

        start_depth = 1
        last_depth = 0

        for node in path[:-1]:
            if node.node_type == NodeType.ROOT:
                task_description = "\n<task_description>\n" + self.root.task_description + "\n</task_description>\n"
                task_description += f"\n<remaining_working_time>\n{hours}hours {minutes}minutes\n</remaining_working_time>\n"
                messages.append(LLMMessage(role="user", content=task_description))
                last_depth = 0
                continue

            if node.node_type == NodeType.SUMMARY:
                summary_messages = self._transform_summary_node_to_react_format(node)
                messages.extend(summary_messages)
                # Remove Turn boundary markers
                last_depth = node.depth
                continue

            if node.node_type == NodeType.REACT:
                react_messages = self._transform_react_node_to_react_format(node)
                messages.extend(react_messages)
                if node.ui_message != "" and node.ui_message is not None:
                    messages.append(LLMMessage(role="user", content=f"<real_user>{node.ui_message}</real_user>"))
                last_depth = node.depth
                continue

        end_depth = last_depth + 1

        self.current_node.set_messages(messages.copy())
        self.current_node.save_to_json(filename_prefix=self.current_node.node_type.value)
        return messages

    def _get_summary_messages(self, depth1: int = None, depth2: int = None) -> List[LLMMessage]:
        """Get message list for SUMMARY node"""
        messages = []
        
        system_prompt = PromptBuilder.build_system_prompt(self.current_node.node_type, prompt_construct_mode="react_default")

        messages.append(LLMMessage(role="system", content=system_prompt))
        messages.append(LLMMessage(role="user", content="\n<task_description>\n" + self.root.task_description + f"\n</task_description>\n<history>\n"))
        current_node = self.current_node
        assert current_node.depth == depth1
        assert len(current_node.parent.children) > 1
        current_node = current_node.parent.children[-2]
        while current_node.depth < depth2:
            assert current_node.node_type == NodeType.REACT or current_node.node_type == NodeType.SUMMARY
            if current_node.node_type == NodeType.REACT:
                messages.extend(self._transform_react_node_to_react_format(current_node))
                if current_node.ui_message != "" and current_node.ui_message != None:
                    messages.append(LLMMessage(role="user", content=f"<real_user>{current_node.ui_message}</real_user>"))
            elif current_node.node_type == NodeType.SUMMARY:
                messages.extend(self._transform_summary_node_to_react_format(current_node))
            # Remove Turn boundary markers
            assert len(current_node.children) > 0
            current_node = current_node.children[-1]
        if messages[-1].role != "user":
            messages.append(LLMMessage(role="user", content=""))
        messages[-1].content = messages[-1].content + "\n</history>\n"
        
        # messages[-1].content = messages[-1].content + "\nPlease reason and generate the summary mentioned in the system prompt."
        # messages[-1].content += f"\n\n **Do not call any tools or functions (e.g., `internal_summarize`). Produce your answer directly following the system prompt in the assistant message content.**\n\n"

        messages[-1].content += f"\n\n **You MUST use the `internal_summarize` action in this turn. No other actions are permitted.**\n\n"

        self.current_node.set_messages(messages.copy())
        self.current_node.save_to_json(filename_prefix=self.current_node.node_type.value)
        return messages
    
    def _get_done_messages(self) -> List[LLMMessage]:
        """Get message list for DONE node"""

        pass

    # Removed: _get_planning_messages (PLANNING related logic)


    # Removed: _get_judge_messages (JUDGE related logic)

    def update_status(self, next_node_type: NodeType, parent_node: BaseNode = None):
        # Create new Node, note that only react nodes can be created without providing parent_node
        new_node = self._create_node(next_node_type, parent_node)
        
        # Update current node
        self.current_node = new_node
        self.total_nodes += 1
        self.current_node.depth = self.current_node.parent.depth + 1

        if new_node.node_type == NodeType.REACT:
            self.calculate_context_stats()
    
    def add_response(self, response: LLMResponse) -> BaseAction:
        """
        Handle LLM response, dispatch to corresponding processing method according to current node type

        Args:
            response: LLM response

        Returns:
            Parsed action
        """
        # Record LLM response
        self.current_step_info["llm_response"] = response
        
        # Dispatch to corresponding processing method according to current node type
        if self.current_node.node_type == NodeType.REACT:
            return self._add_react_response(response)
        elif self.current_node.node_type == NodeType.SUMMARY:
            return self._add_summary_response(response)
        elif self.current_node.node_type == NodeType.DONE:
            return self._add_done_response(response)
        else:
            raise ValueError(f"Unsupported node type: {self.current_node.node_type}")
    
    def _add_react_response(self, response: LLMResponse) -> BaseAction:
        """Handle LLM response for REACT node"""
        try:
            # Extract thought content
            # content = response.content if hasattr(response, 'content') else str(response)
            # thought = self._extract_thought(content)
            
            # Convert tool call to action
            action, response = self._convert_tool_call_to_action(response)
            
            # Set response
            self.current_node.set_response(response)
            
            # Set thought and action
            # if thought:
            #     self.current_node.set_thought(thought)
            self.current_node.set_action(action)
            
            print("_add_react_response -> action: ", action)
            
            # Record output actions
            self.current_step_info["output_actions"] = [action]
            
            return action
            
        except Exception as e:
            raise e
    
    def _add_summary_response(self, response: LLMResponse) -> BaseAction:
        """Handle LLM response for SUMMARY node"""
        tool_calls = response.tool_calls
        tool_call = tool_calls[0] if tool_calls and len(tool_calls) > 0 else None
        if not tool_call:
            # If no explicit action, use NULL by default
            print(f"InternalSummarize is not called in SUMMARY node")
            return False
        print("_add_summary_response -> action: ", tool_call)
        arguments = tool_call.arguments
        
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        
        summary_content = arguments.get("summary_content", "")
        add_response_result = self.current_node.set_summary(response, summary_content)
        return add_response_result["success"]
    
    def _add_done_response(self, response: LLMResponse) -> BaseAction:
        """Handle LLM response for DONE node"""
        pass

    # Removed: _add_judge_response and _add_planning_response (JUDGE/PLANNING related processing)

    def add_observation(self, observation: BaseObservation = None, cli_message: str = None, ui_message: str = None):
        """
        Add observation result to current node

        Args:
            observation: Observation result (usually BaseObservation)
            cli_message: CLI message
            ui_message: UI message
        """
        # Record input information
        self.current_step_info["input_observation"] = observation
        self.current_step_info["input_cli_message"] = cli_message
        self.current_step_info["input_ui_message"] = ui_message
        
        # If current node is ReactNode, set observation result
        if isinstance(self.current_node, ReActNode):
            self.current_node.set_observation(observation, cli_message, ui_message)
            # self.update_total_tokens_by_observation(observation)
        self.logger.debug(f"Add observation result to node {self.current_node.node_type.value}")
    
    def _convert_tool_call_to_action(self, response: LLMResponse) -> Tuple[BaseAction, LLMResponse]:
        """
        Convert tool call to action
        """
        tool_calls = response.tool_calls
        tool_call = tool_calls[0] if tool_calls and len(tool_calls) > 0 else None
        if not tool_call:
            # If no explicit action, use NULL by default
            action = get_action_class(ActionType.NULL)(
                call_id=make_toolcall_id(), 
                error_message=f"Do nothing. Please check your output format. Do not output more than 3000 token."
            )
            tool_call = ToolCall(call_id=action.call_id, name=ActionType.NULL.value)
            response.tool_calls = []
            response.tool_calls.append(tool_call)
            return action, response
        
        arguments = tool_call.arguments
        
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        if len(tool_call.name) >= 64:
            tool_call.name = tool_call.name[:64]

        action_class = get_action_class(tool_call.name)
        if action_class:
            try:
                action = action_class(**tool_call.arguments)
                action.call_id = tool_call.call_id
            except Exception as e:
                action = get_action_class(ActionType.NULL)(
                    call_id=tool_call.call_id,
                    error_message=f"Tool `{tool_call.name}` occurs error: {e}"
                )
        else:
            if not re.match(r'^[a-zA-Z0-9_-]+$', response.tool_calls[0].name):
                action = get_action_class(ActionType.NULL)(
                    call_id=tool_call.call_id,
                    error_message=f'Unknown tool: `{tool_call.name}`. The tool name must satisfy regular expression pattern: "[a-zA-Z0-9_-]+".'
                )
                response.tool_calls[0].name = re.sub(r'[^a-zA-Z0-9_-]', '_', response.tool_calls[0].name)
            else:
                # If there is no explicit action, use NULL by default
                action = get_action_class(ActionType.NULL)(
                    call_id=tool_call.call_id,
                    error_message=f"Unknown tool: `{tool_call.name}`."
                )
        return action, response
    
    def should_summarize(self) -> bool:
        """
        Determine whether summarization is needed

        Returns:
            Whether summarization is needed
        """
        
        if self.context_stats["total_tokens"] >= self.limits.summary_threshold or self.context_stats["total_tokens"] - self.limits.max_tokens  >= self.limits.context_length:
            return True
        
        return False

    # Removed: has_planning_node_in_path (PLANNING count logic)

    def _create_node(self, node_type: NodeType,  parent_node: BaseNode = None) -> BaseNode:
        """Create new node"""
        # Check if node type is valid
        if not self._is_valid_node_type(node_type):
            raise ValueError(f"Invalid node type: {node_type}")
        

        if parent_node is not None:
            
            if not isinstance(parent_node, BaseNode):
                raise ValueError(f"Invalid parent node: {parent_node}")
            
            try:
                # TODO: can be changed to a location together with agent_main.log
                parent_node.save_to_json(filename_prefix=parent_node.node_type.value)
            except Exception as e:
                print(f"Warning: failed to save current node JSON: {e}")
            
            if node_type == NodeType.REACT:
                new_node = ReActNode(parent_node)
            elif node_type == NodeType.SUMMARY:
                new_node = SummaryNode(parent_node)
            else:
                raise ValueError(f"Invalid parent node: {parent_node}")
            return new_node
        else:
            try:
                # TODO: can be changed to a location together with agent_main.log
                self.current_node.save_to_json(filename_prefix=self.current_node.node_type.value)
            except Exception as e:
                print(f"Warning: failed to save current node JSON: {e}")
                
            if node_type == NodeType.REACT:
                new_node = ReActNode(self.current_node)
            else:
                raise ValueError(f"Invalid parent node: {parent_node}")
            return new_node
        
    def calculate_context_stats(self) -> Dict[str, Any]:
        
        """
        Calculate context statistics

        Returns:
            Dictionary containing statistics
        """
        if not self.current_node:
            return self.context_stats
        
        total_tokens = 0
        
        system_prompt = PromptBuilder.build_system_prompt(self.current_node.node_type, prompt_construct_mode="react_default")
        
        total_tokens += self.cal_prompt_tokens(system_prompt)

        # TODO: directly build a prompt according to the longest possibility, return an estimated value of context
        path = self.current_node.get_path_from_root()
        for node in path[:-1]:
            if node.node_type == NodeType.ROOT:
                task_description = "\n<task_description>\n" + self.root.task_description + "\n</task_description>\n"
                total_tokens += self.cal_prompt_tokens(task_description)
                continue

            if node.node_type == NodeType.SUMMARY:
                summary_messages = self._transform_summary_node_to_react_format(node)
                for summary_message in summary_messages:
                    total_tokens += self.cal_prompt_tokens(summary_message.content)
                continue

            if node.node_type == NodeType.REACT:
                react_messages = self._transform_react_node_to_react_format(node)
                for react_message in react_messages:
                    total_tokens += self.cal_message_tokens(react_message)
                continue

        self.context_stats["total_tokens"] = total_tokens + 1000
        self.context_stats["node_count"] = len(path[:-1])

        print(f"context_stats: {self.context_stats}")

        return self.context_stats

    def check_internal_actions_validation(self, action_type: ActionType, depth1: int = None, depth2: int = None) -> Dict[str, Any]:
        """
        Check if internal action is valid, if valid return True and parent node, if invalid return False and reason
        """
        if action_type == ActionType.SUMMARIZE:
            if isinstance(depth1, int) and isinstance(depth2, int):
                if depth1 >= depth2:
                    reason = "The depth1 must be less than or equal to depth2."
                    parent_node = None
                    valid = False
                else:
                    current_node = self.current_node.parent
                    if current_node.depth < depth2 - 1:
                        reason = "The depth2 must be in the history."
                        parent_node = None
                        valid = False
                    else:
                        while current_node.depth > depth1:
                            current_node = current_node.parent
                        if current_node.depth == depth1 and current_node.parent:
                            reason = ""
                            parent_node = current_node.parent
                            valid = True
                        else:
                            reason = "The depth1 must be in the history."
                            parent_node = None
                            valid = False
            else:
                reason = "The depth1 and depth2 must be integers and greater than 0."
                parent_node = None
                valid = False

        elif action_type == ActionType.THINK:
            parent_node = self.current_node
            # Only allow use in REACT node
            if self.current_node.node_type != NodeType.REACT:
                reason = "ThinkAction can only be used in REACT node."
                valid = False
            # Continuous use restriction: if the last round just used Think, then this round is not allowed
            elif self.think_cooldown:
                reason = "ThinkAction was used in the last round. You cannot use it in consecutive rounds, try another tools."
                valid = False
            else:
                reason = ""
                valid = True
            
        else:
            raise ValueError(f"Invalid action type: {action_type}")
        return {
            "is_valid": valid,
            "reason": reason,
            "parent_node": parent_node
        }
    
    def _transform_react_node_to_react_format(self, react_node: ReActNode) -> List[LLMMessage]:

        # TODO: if it's a generated react node, also need to show the depth of each node
        messages = []
        tool_calls = react_node.response.tool_calls
        tool_call = tool_calls[0] if tool_calls and len(tool_calls) > 0 else None
        
        assistant_message = LLMMessage(
            role=react_node.response.role,
            content=react_node.response.content,
            # reasoning=react_node.response.reasoning,
            tool_call=tool_call,
        )
        messages.append(assistant_message)
        # messages.append(LLMMessage(role="assistant", content=react_node.observation.message))
        if react_node.observation is not None and react_node.observation.tool_name is not None:
            tool_result = ToolResult(
                name=react_node.observation.tool_name.value,
                call_id=react_node.observation.tool_call_id,
                content=json.dumps(react_node.observation.to_dict()),
            )
            tool_message = LLMMessage(role="tool", tool_result=tool_result, content="")
            messages.append(tool_message)

        return messages
    
    def _transform_summary_node_to_react_format(self, summary_node: SummaryNode) -> List[LLMMessage]:
        messages = []
        summary_content = ""
        if summary_node.summary_content:
            summary_content = summary_node.summary_content
        else:
            print("summary_content is empty! Use 'The summary content is EMPTY.' ")
            summary_content = "The summary content is EMPTY."
        depth1 = summary_node.depth
        node = summary_node.parent.children[-2]
        while isinstance(node.children,list) and len(node.children) > 0:
            node = node.children[-1]
        depth2 = node.depth
        depth = depth2 - depth1

        summary_content = f"""After the last action, I have also made {depth} actions and get {depth} observations.
However, the context about the these actions and observations is too long. So I just summarize them. 
Here is the summary that maybe useful in the future about these actions and observations:

{summary_content}
"""
        messages.append(LLMMessage(role="assistant", content=summary_content))
        messages.append(LLMMessage(role="user", content="OK, continue to finish your task based on the history.  You should using `list_sessions`, `check_session_idle` and `get_session_output`to list and check All session status first! If the session output is empty, do not check it twice"))
        return messages
