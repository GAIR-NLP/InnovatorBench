"""
Summary node class - Information summarization and compression
"""

from typing import List, Optional, Dict, Any
from dataclasses import asdict
from agents.context.nodes import BaseNode, NodeType
from llm.llm_basics import LLMMessage, LLMResponse
from research_gym.schema import ActionType

class SummaryNode(BaseNode):
    """Summary node"""
    def __init__(self, parent: Optional[BaseNode] = None):
        super().__init__(parent, NodeType.SUMMARY)
        self.summary_content: str = ""
    
    def set_summary(self, summary: LLMResponse, summary_content: str):
        """Set summary content"""
        self.response = summary
        self.summary_content = summary_content
        self.save_to_json(BaseNode.directory, filename_prefix=self.node_type.value)
        return {
            "success": True,
            "message": "Summary set successfully"
        }
    
    def duplicate(self, parent: BaseNode = None) -> BaseNode:
        """Duplicate node"""
        parent = self.parent if parent is None else parent
        new_node = SummaryNode(parent)
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
        new_node.summary_content = self.summary_content
        self.save_to_json(BaseNode.directory, filename_prefix=self.node_type.value)
        return new_node

    def _to_dict_content(self) -> Dict[str, Any]:
        """Convert node to dictionary."""
        d = super()._to_dict_content()
        d["summary_content"] = self.summary_content
        return d
    
    def _from_dict(self, data: Dict[str, Any]):
        super()._from_dict(data)
        self.summary_content = data.get("summary_content", "")