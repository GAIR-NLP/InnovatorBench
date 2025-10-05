from typing import List, Optional, Dict, Any
from research_gym.action.action import BaseAction
from research_gym.observation.observation import BaseObservation
from research_gym.action.handlers import (
    ActionHandler, CommandActionHandler, FileActionHandler, ParseActionHandler, ThinkingActionHandler, EvalActionHandler, ViewHintActionHandler, SleepActionHandler, SearchActionHandler, WebActionHandler
)
from research_gym.applications.cmd_operations import CmdOperations
from research_gym.applications.file_operations import FileOperations
from research_gym.applications.parser_operations import ParserOperations

from research_gym.applications.search_operations import SearchOperations

from research_gym.applications.web_operations import WebOperations

from research_gym.configs.task_config import TaskConfig
import logging
import asyncio


class ActionManager:
    """Action manager - using factory pattern and chain of responsibility pattern"""
    
    def __init__(self, task_config: TaskConfig):
        self.task_config = task_config
        self.handlers: List[ActionHandler] = []
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Operations classes for lazy initialization
        self._cmd_operations = None
        self._file_operations = None
        self._parser_operations = None
        self._search_operations = None
        self._web_operations = None
        
    def _ensure_operations_initialized(self):
        """Ensure all operations classes are initialized - using lazy initialization"""
        if self._cmd_operations is None:
            self._cmd_operations = CmdOperations(
                default_shell=self.task_config.default_shell,
                default_http_port=self.task_config.default_http_port,
                proxy_url=self.task_config.cmd_proxy_url,  # Use cmd_proxy_url
                env_vars=self.task_config.env_vars
            )
        
        if self._file_operations is None:
            self._file_operations = FileOperations(self.task_config.workspace)
        
        if self._parser_operations is None and ParserOperations is not None:
            self._parser_operations = ParserOperations(
                openai_api_key=self.task_config.openai_api_key,
                openai_base_url=self.task_config.openai_base_url,
            )
        
        if self._search_operations is None and SearchOperations is not None:
            try:
                # Use search configuration from TaskConfig
                search_config = {
                    'search_engine': self.task_config.search_engine,
                    'serper_api_key': self.task_config.serper_api_key,
                    'azure_bing_search_subscription_key': self.task_config.azure_bing_search_subscription_key,
                    'search_max_top_k': self.task_config.search_max_top_k,
                    'search_region': self.task_config.search_region,
                    'search_lang': self.task_config.search_lang,
                    'azure_bing_search_mkt': self.task_config.azure_bing_search_mkt
                }
                self._search_operations = SearchOperations(
                    config=search_config,
                    cache_dir=self.task_config.search_cache_dir,
                    cache_duration_days=self.task_config.search_cache_duration_days
                )
            except (ValueError, Exception) as e:
                self.logger.warning(f"Search function initialization failed, search function will be unavailable: {e}")
                self._search_operations = None
        
        if self._web_operations is None and WebOperations is not None:
            try:
                # Use Web configuration from TaskConfig
                self._web_operations = WebOperations(
                    server_host=self.task_config.web_server_host,
                    server_port=self.task_config.web_server_port,
                    proxy_url=self.task_config.web_proxy_url if self.task_config.web_proxy_url else None,
                    cache_dir=self.task_config.web_cache_dir,
                    cache_duration_days=self.task_config.web_cache_duration_days
                )
            except Exception as e:
                self.logger.warning(f"Web browsing function initialization failed, browsing function will be unavailable: {e}")
                self._web_operations = None
    
    def _create_handlers(self) -> List[ActionHandler]:
        """Factory method: create all handlers"""
        self._ensure_operations_initialized()
        
        handlers = [
            CommandActionHandler(self._cmd_operations),
            FileActionHandler(self._file_operations),
            ThinkingActionHandler(),
            ViewHintActionHandler(self.task_config),
            EvalActionHandler(self.task_config),
            SleepActionHandler(),  # Add sleep handler
        ]
        
        if self._parser_operations is not None:
            handlers.append(ParseActionHandler(self._parser_operations))
        
        # Add search handler
        if self._search_operations is not None:
            handlers.append(SearchActionHandler(self._search_operations))
        
        # Add web browsing handler  
        if self._web_operations is not None:
            handlers.append(WebActionHandler(self._web_operations))
        
        return handlers
    
    def initialize(self):
        """Initialize handler chain"""
        self.handlers = self._create_handlers()
        self.logger.info(f"Initialized {len(self.handlers)} action handlers")
    
    def execute_action(self, action: BaseAction) -> BaseObservation:
        """Execute action - using chain of responsibility pattern"""
        if not self.handlers:
            self.initialize()
        
        # Chain of responsibility pattern: try each handler in sequence
        for handler in self.handlers:
            if handler.can_handle(action):
                self.logger.debug(f"Using {handler.__class__.__name__} to handle {action.__class__.__name__}")
                return handler.handle(action)
        
        # If no handler can handle this action
        error_msg = f'No handler found to handle {type(action).__name__}'
        self.logger.error(error_msg)
        return self._create_error_observation(error_msg)
    
    def _create_error_observation(self, message: str) -> BaseObservation:
        """Create error observation result"""
        from research_gym.action.empty import NullAction
        null_action = NullAction()
        return BaseObservation.from_result({
            'success': False,
            'message': message,
            'output': {'output': []}
        }, null_action)
    
    def get_supported_actions(self) -> Dict[str, List[str]]:
        """Get list of supported action types"""
        if not self.handlers:
            self.initialize()
        
        supported_actions = {}
        for handler in self.handlers:
            handler_name = handler.__class__.__name__
            # This is a simplified implementation, actual scenarios may need more complex logic
            if hasattr(handler, 'action_types'):
                supported_actions[handler_name] = [
                    action_type.__name__ for action_type in handler.action_types
                ]
        
        return supported_actions
    
    def add_handler(self, handler: ActionHandler):
        """Add custom handler"""
        self.handlers.append(handler)
        self.logger.info(f"Added handler: {handler.__class__.__name__}")
    
    def remove_handler(self, handler_class: type):
        """Remove handler of specified type"""
        original_count = len(self.handlers)
        self.handlers = [h for h in self.handlers if not isinstance(h, handler_class)]
        removed_count = original_count - len(self.handlers)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} {handler_class.__name__} handlers")
    
    def clear_handlers(self):
        """Clear all handlers"""
        self.handlers.clear()
        self.logger.info("Cleared all handlers")
    
    def get_handler_stats(self) -> Dict[str, Any]:
        """Get handler statistics"""
        if not self.handlers:
            self.initialize()
        
        stats = {
            'total_handlers': len(self.handlers),
            'handler_types': [handler.__class__.__name__ for handler in self.handlers],
            'operations_initialized': {
                'cmd_operations': self._cmd_operations is not None,
                'file_operations': self._file_operations is not None,
                'parser_operations': self._parser_operations is not None
            }
        }
        
        return stats
    
    def parallel_tool_call(self, tool_calls: list[BaseAction]) -> list[BaseObservation]:
        """Execute tool calls in parallel"""
        # return await asyncio.gather(*[self.execute_action(call) for call in tool_calls])
        return [self.execute_action(call) for call in tool_calls]

    def sequential_tool_call(self, tool_calls: list[BaseAction]) -> list[BaseObservation]:
        """Execute tool calls in sequential"""
        return [self.execute_action(call) for call in tool_calls]  
    
    # async def parallel_tool_call(self, tool_calls: list[BaseAction]) -> list[BaseObservation]:
    #     """Execute tool calls in parallel"""
    #     return await asyncio.gather(*[self.execute_action(call) for call in tool_calls])

    # async def sequential_tool_call(self, tool_calls: list[BaseAction]) -> list[BaseObservation]:
    #     """Execute tool calls in sequential"""
    #     return [await self.execute_action(call) for call in tool_calls]