"""
Context management module - Public interface exports

Provide imports for all public classes and functions, maintain backward compatibility
"""

# Type definitions and configuration
from ..config.types import ContextLimits

# Decision makers and builders
from .prompt_builder import PromptBuilder

# For backward compatibility, export all important classes
__all__ = [
    # Types and configuration
    'NodeType',
    'ContextLimits',
    
    # Node classes
    'BaseNode',
    'ReActNode',
    'SummaryNode',
    
    # Utility classes
    'PromptBuilder',
    
    # Main managers
    'BaseManager',
    'ContextManager',
    'ReActManager',
] 