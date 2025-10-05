"""
Compatibility module, provide unified typing support for different Python versions
"""

import sys
from typing import Any, Callable, TypeVar

# Python version check
if sys.version_info >= (3, 12):
    from typing import override
else:
    # Provide stub implementation of override decorator for Python 3.10/3.11
    def override(func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorator, mark method as overriding parent class method
        In Python versions before 3.12, this is just an empty decorator
        """
        return func

# Export all required types
__all__ = ["override"] 