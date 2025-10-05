"""
Context manager base class - Define core interface for context management
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from datetime import datetime
import logging
import json
import os
from transformers import AutoTokenizer
from agents.context.nodes.base_node import BaseNode, NodeType
from agents.context.nodes import ReActNode, SummaryNode
from agents.config.types import ContextLimits
from llm.config import AgentConfig
from research_gym.configs.task_config import TaskConfig
from llm.llm_basics import LLMMessage, LLMResponse
from research_gym.action import BaseAction
from research_gym.observation.observation import BaseObservation

class BaseManager(ABC):
    """
    Context manager base class - Define core interface for context management

    This abstract base class defines the core methods that all context managers must implement,
    providing a unified interface for different types of context management strategies.
    """
    
    def __init__(self, agent_config: AgentConfig, task_config: TaskConfig, node_types: Optional[List[NodeType]] = None):
        """
        Initialize base manager

        Args:
            agent_config: Agent configuration
            task_config: Task configuration
            node_types: Node types
        """
        self.root: BaseNode = None
        self.current_node: BaseNode = None
        self.limits = agent_config.context_limits

        self.node_types = node_types if node_types is not None else self._get_default_node_type()
        self.task_config = task_config

        self.tokenizer = AutoTokenizer.from_pretrained(self.task_config.tokenizer)
        
        # Statistics information
        self.total_nodes = 0
        self.context_stats = {
            "total_tokens": 0,
            "node_count": 0,
            "max_tokens": self.limits.max_tokens,
            "summary_threshold": self.limits.summary_threshold
        }
        
        # Current step record information
        self.current_step_info: Dict[str, Any] = {
            "input_observation": None,
            "input_cli_message": None,
            "input_ui_message": None,
            "llm_messages": [],
            "llm_response": None,
            "output_actions": [],
            "error": None
        }
        
        # Remaining time
        self.start_time = self.task_config.start_time

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # Call specific implementation initialization method
        self._initialize_context()
    
    @abstractmethod
    def _get_default_node_type(self) -> List[NodeType]:
        """
        Get default node types
        """
        pass

    @abstractmethod
    def _initialize_context(self):
        """
        Initialize specific context structure
        """
        pass

    def _is_valid_node_type(self, node_type: NodeType) -> bool:
        """
        Check if node type is valid
        """
        return node_type in self.node_types
    
    @abstractmethod
    def add_observation(self, observation: BaseObservation, cli_message: str = None, ui_message: str = None):
        """
        Add observation result to current node

        Args:
            observation: Observation result
            cli_message: CLI message
            ui_message: UI message
        """
        pass
    
    @abstractmethod
    def get_messages(self) -> List[LLMMessage]:
        """
        Build message list to send to LLM

        Returns:
            Formatted message list
        """
        pass
    
    @abstractmethod
    def add_response(self, response: LLMResponse) -> BaseAction:
        """
        Handle LLM response, create new node or update state

        Args:
            response: LLM response

        Returns:
            Parsed action
        """
        pass
    
    @abstractmethod
    def _create_node(self, node_type: NodeType, response: LLMResponse) -> BaseNode:
        """Create new node"""
        pass
    
    @abstractmethod
    def update_status(self):
        """
        Update context manager state
        """
        pass
    
    @abstractmethod
    def calculate_context_stats(self) -> Dict[str, Any]:
        """
        Calculate context statistics

        Returns:
            Dictionary containing statistics
        """
        pass
        
    def cal_response_tokens(self, response: LLMResponse) -> int:
        total_tokens = 0
        if response.content:
            total_tokens += len(self.tokenizer.encode(response.content))
        if response.reasoning:
            total_tokens += len(self.tokenizer.encode(response.reasoning))
        if response.tool_calls:
            for tc in response.tool_calls:
                total_tokens += len(self.tokenizer.encode(tc.name))
                if tc.arguments:
                    if isinstance(tc.arguments, dict):
                        total_tokens += len(self.tokenizer.encode(json.dumps(tc.arguments)))
                    else:
                        total_tokens += len(self.tokenizer.encode(str(tc.arguments)))
        return total_tokens

    def cal_observation_tokens(self, observation: BaseObservation) -> int:
        total_tokens = 0
        if observation.tool_call_id:
            total_tokens += len(self.tokenizer.encode(observation.tool_name.value))
            total_tokens += len(self.tokenizer.encode(observation.tool_call_id))
            total_tokens += len(self.tokenizer.encode(json.dumps(observation.to_dict())))
        return total_tokens
    
    def cal_message_tokens(self, message: LLMMessage) -> int:
        total_tokens = 0
        if message.content:
            total_tokens += len(self.tokenizer.encode(message.content))
        if message.tool_call:
            # Count tool_call function name and arguments
            total_tokens += len(self.tokenizer.encode(message.tool_call.name))
            if message.tool_call.arguments:
                if isinstance(message.tool_call.arguments, dict):
                    total_tokens += len(self.tokenizer.encode(json.dumps(message.tool_call.arguments)))
                else:
                    total_tokens += len(self.tokenizer.encode(str(message.tool_call.arguments)))
        if message.tool_result:
            total_tokens += len(self.tokenizer.encode(message.tool_result.name))
            total_tokens += len(self.tokenizer.encode(message.tool_result.call_id))
            total_tokens += len(self.tokenizer.encode(message.tool_result.content))
        return total_tokens
    
    def cal_prompt_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt))
    
    def record_agent_step(self, reflection: Optional[str] = None):
        """
        Record complete agent execution step

        Args:
            reflection: Agent's reflection on this step (optional)
        """
        
        # Build complete step record
        step_record = {
            # Basic information
            "timestamp": datetime.now().isoformat(),
            "state": "completed" if not self.current_step_info.get("error") else "error",
            
            # Input information
            "input": {
                "observation": self.current_step_info.get("input_observation"),
                "cli_message": self.current_step_info.get("input_cli_message"),
                "ui_message": self.current_step_info.get("input_ui_message")
            },
            
            # LLM interaction information
            "llm_interaction": {
                "messages": self.current_step_info.get("llm_messages", []),
                "response": self.current_step_info.get("llm_response")
            },
            
            # Output information
            "output": {
                "actions": self.current_step_info.get("output_actions", [])
            },
            
            # Current node information
            "current_node": self._get_current_node_info(),
            
            # Context manager state
            "context_manager_state": {
                "total_nodes": self.total_nodes,
                "context_stats": self.context_stats,
                "limits": {
                    "max_tokens": self.limits.max_tokens,
                    "summary_threshold": self.limits.summary_threshold,
                }
            },
            
            # Other information
            "reflection": reflection,
            "error": self.current_step_info.get("error")
        }
    
    def _get_current_node_info(self) -> Dict[str, Any]:
        """Get detailed information of current node"""
        if not self.current_node:
            return {}
        
        node_info = {
            "node_id": id(self.current_node),
            "node_type": self.current_node.node_type.value,
            "parent_id": id(self.current_node.parent) if self.current_node.parent else None,
            "quality": self.current_node.quality,
            "metadata": self.current_node.metadata.copy()
        }
        
        # Add node specific information
        node_info.update(self._get_node_specific_info())
        
        return node_info
    
    def _get_node_specific_info(self) -> Dict[str, Any]:
        """
        Get information of specific node type
        Subclasses can override this method to add information for specific node types
        """
        return {}
    
    def get_information_for_user(self) -> Dict[str, Any]:
        """
        Get user information

        Returns:
            Dictionary containing information of interest to users
        """
        info = {
            "current_node_type": self.current_node.node_type.value if self.current_node else "UNKNOWN",
            "total_nodes": self.total_nodes,
        }
        
        # Add current node's response information
        if self.current_node and self.current_node.response:
            info.update({
                "thought": self.current_node.response.content,
                "raw_response": self.current_node.response.content,
            })
        
        # Add context statistics
        # self.calculate_context_stats()
        info.update({
            "context_tokens": self.context_stats["total_tokens"],
            "context_nodes": self.context_stats["node_count"]
        })
        
        return info
    
    def _to_dict(self) -> Dict[str, Any]:
        """
        Convert context manager to dictionary
        """
        return {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert context manager to dictionary
        """
        d = self._to_dict()
        d["total_nodes"] = self.total_nodes
        d["context_stats"] = self.context_stats
        d["limits"] = asdict(self.limits)
        d["node_types"] = [nt.value for nt in self.node_types]
        d["start_time"] = self.start_time.isoformat()
        return d

    def from_dict(self, d: Dict[str, Any]):
        """
        Restore context manager from dictionary
        """
        self.start_time = datetime.fromisoformat(d["start_time"])
        self.total_nodes = d["total_nodes"]
        self.context_stats = d["context_stats"]
        self.limits = ContextLimits(**d["limits"])
        self.node_types = [NodeType(nt) for nt in d["node_types"]]
    
    def load_from_tree(self, tree_path: Optional[str] = None) -> bool:
        """
        Read conversation tree from saved tree_data.json, rebuild as Node tree (using from_dict/_from_dict of each node class),
        and jump current_node to the latest "complete" node (with non-empty messages and response).
        """
        try:
            # 1) Parse path
            if tree_path is None:
                workspace_dir = getattr(self.task_config, "workspace", "") or ""
                tree_path = os.path.join(workspace_dir, "tree_data.json")
            if not tree_path or not os.path.exists(tree_path):
                return False

            # 2) Read JSON
            with open(tree_path, "r", encoding="utf-8") as f:
                tree_dict: Dict[str, Any] = json.load(f)

            # 3) Build tool method
            latest_complete_node_ref: Optional[BaseNode] = None
            latest_complete_ts: Optional[datetime] = None
            total_nodes_count: int = 0

            def _is_complete(node: BaseNode) -> bool:
                has_messages = bool(getattr(node, "messages", None))
                resp = getattr(node, "response", None)
                has_response = bool(resp and getattr(resp, "content", None))
                return has_messages and has_response

            def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
                if not ts:
                    return None
                try:
                    return datetime.fromisoformat(ts)
                except Exception:
                    return None

            def _new_node(ntype: NodeType, parent: Optional[BaseNode]) -> BaseNode:
                if ntype == NodeType.ROOT:
                    node = ReActNode(parent)
                    node.node_type = NodeType.ROOT
                    return node
                elif ntype == NodeType.REACT:
                    return ReActNode(parent)
                if ntype == NodeType.SUMMARY:
                    return SummaryNode(parent)
                # DONE / Unknown type -> Use BaseNode to carry
                return BaseNode(parent=parent, node_type=ntype)

            def _build(d: Dict[str, Any], parent: Optional[BaseNode] = None) -> BaseNode:
                nonlocal latest_complete_node_ref, latest_complete_ts, total_nodes_count

                node_type_str = str(d.get("node_type", "react") or "react")
                try:
                    ntype = NodeType(node_type_str)
                except Exception:
                    ntype = NodeType.REACT  # Compatible with historical values

                # Create the corresponding node instance according to the type (will be automatically hung on parent.children)
                node = _new_node(ntype, parent)

                # Use node class's own from_dict/_from_dict to fill fields (not processing children)
                node.from_dict(d)

                total_nodes_count += 1
                if _is_complete(node):
                    cur_ts = _parse_ts(getattr(node, "timestamp", None))
                    if cur_ts is None:
                        latest_complete_node_ref = node
                    else:
                        if latest_complete_ts is None or cur_ts >= latest_complete_ts:
                            latest_complete_ts = cur_ts
                            latest_complete_node_ref = node

                # Recursively build children (note: from_dict will not process children)
                for child_dict in d.get("children", []) or []:
                    _build(child_dict, parent=node)

                return node

            # 4) Rebuild the entire tree
            root = _build(tree_dict, parent=None)

            # 5) Cover to manager
            self.root = root
            self.current_node = latest_complete_node_ref or root

            # 6) Update statistics
            try:
                self.total_nodes = total_nodes_count
                if hasattr(self, "context_stats") and isinstance(self.context_stats, dict):
                    self.context_stats["node_count"] = total_nodes_count
            except Exception:
                pass

            return True
        except Exception as e:
            self.logger.error(f"load_from_tree failed: {e}")
            raise
    
    