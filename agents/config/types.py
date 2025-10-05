"""
Context management type definitions and configuration
"""

from enum import Enum
from dataclasses import dataclass

class TaskStatus(Enum):
    """Task status enumeration"""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ContextLimits:
    """Context length limit configuration"""
    max_tokens: int = 4096  # Remaining token count
    summary_threshold: int = 100000 # 100000  # Increase summary threshold
    context_length: int = 128000  # Model's actual context length
    max_internal_action_times: int = -1  # Maximum number of times internal actions can be executed, if -1 then no limit, temporarily deprecated, default is -1
