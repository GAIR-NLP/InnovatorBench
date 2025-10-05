"""
React node class - Handle think-act-observe loop
"""

import json
from typing import List, Optional, Dict, Any

from dataclasses import asdict
from agents.context.nodes import BaseNode, NodeType
from research_gym.action import BaseAction, ToolCall, ToolResult, get_action_class, NullAction
from research_gym.observation import BaseObservation
from research_gym.observation.observation import ObservationFactory
from research_gym.schema.observation import ObservationType
from llm.llm_basics import LLMMessage, LLMResponse
from research_gym.schema import ActionType
from research_gym.observation import get_observation_class
from research_gym.base_node import Source
from datetime import datetime


class ReActNode(BaseNode):
    """ReAct execution node"""
    def __init__(self, parent: Optional[BaseNode] = None):
        super().__init__(parent, NodeType.REACT)
        self.observation: BaseObservation = None
        self.cli_message: str = None
        self.ui_message: str = None
        self.action: BaseAction = None
        # self.thought: str = None
    
    def set_observation(self, observation: BaseObservation = None, cli_message: str = None, ui_message: str = None):
        """Set observation result"""
        self.observation = observation
        self.cli_message = cli_message
        self.ui_message = ui_message
        self.save_to_json(BaseNode.directory, filename_prefix=self.node_type.value)
    
    def set_action(self, action: BaseAction):
        """Set action"""
        self.action = action
        self.save_to_json(BaseNode.directory, filename_prefix=self.node_type.value)
    
    def set_task_description(self, task_description):
        """Set task description"""
        assert self.node_type == NodeType.ROOT, "Only ROOT node can set task description"
        self.task_description = task_description
        self.save_to_json(BaseNode.directory, filename_prefix=self.node_type.value)
    
    def duplicate(self, parent: BaseNode = None) -> BaseNode:
        """Duplicate node"""
        # TODO: Change to only duplicate right child
        parent = self.parent if parent is None else parent
        new_node = ReActNode(parent)
        new_node.depth = parent.depth+1
        new_node.messages = self.messages.copy()
        new_node.response = self.response
        new_node.add_source("duplicate")
        if len(self.children) > 0:
            if self.children[-1].node_type == NodeType.SUMMARY:
                new_node.children = [self.children[-1].duplicate(parent=new_node)]
            elif self.children[-1].node_type == NodeType.REACT:
                if self.children[-1].action.action_type != ActionType.SUMMARIZE:
                    new_node.children = [self.children[-1].duplicate(parent=new_node)]
            else:
                raise ValueError("Invalid node type in duplicate")
        # new_node.metadata = self.metadata.copy()
        
        new_node.observation = self.observation
        new_node.action = self.action
        new_node.ui_message = self.ui_message
        new_node.cli_message = self.cli_message
        self.save_to_json(BaseNode.directory, filename_prefix=self.node_type.value)
        return new_node
    
    def _to_dict_content(self) -> Dict[str, Any]:
        """Convert node to dictionary."""
        d = super()._to_dict_content()
        d.update({
            "observation": self.observation.to_dict() if self.observation else None,
            "action": asdict(self.action) if self.action else None,
            # "ui_message": self.ui_message,
            # "cli_message": self.cli_message,
        })
        return d
    
    def _from_dict(self, data: Dict[str, Any]):
        super()._from_dict(data)
        if NodeType(data["node_type"]) == NodeType.ROOT:
            self.task_description = data["observation"]["message"]
            obs = BaseObservation()
            obs.success = data["observation"]["success"]
            obs._message = data["observation"]["message"]
            obs._source = Source(data["observation"]["source"])
            obs.timestamp = datetime.fromisoformat(data["observation"]["timestamp"])
            self.observation = obs
        else:
            action = None
            if data["action"]:
                action_type = data["action"]["action_type"]
                data["action"].pop("action_type")
                self.action = get_action_class(ActionType(action_type))(**data["action"])
                action = self.action
            assert action is not None, "Action is None"
            if data["observation"]:
                self.observation = ObservationFactory.create_observation_by_type(
                    ObservationType(data["observation"]["observation_type"]),
                    data["observation"],
                    action
                )
        # self.ui_message = data.get("ui_message", None)
        # self.cli_message = data.get("cli_message", None)