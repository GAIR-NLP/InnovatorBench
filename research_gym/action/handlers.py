import os
import sys
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable, Dict, Any, Type, TypeVar, Union
from datetime import datetime
from importlib import import_module

from research_gym.action.action import BaseAction
from research_gym.action import NullAction
# Import specific Action types
from research_gym.action.commands import (
    CommandBaseAction, CreateSessionAction, ListSessionsAction, RunCommandAction,
    InputInSessionAction, GetSessionOutputAction, GetSessionRecentOutputAction,
    SessionStatusAction, SessionIdleAction, ClearSessionBufferAction,
    CloseSessionAction, CloseAllSessionsAction, KillSessionProcessesAction
)

from research_gym.action.files import (
    FileEditAction, OpenFileAction,
    GotoLineAction, FileScrollDownAction, FileScrollUpAction, CreateFileAction,
    SearchDirAction, SearchFileAction, FindFileAction, ListFilesAction,
    GetFileInfoAction
)

from research_gym.action.system import FinishAction, EvalAction, ViewHintAction, ThinkAction

from research_gym.action.parses import (
    ParsePdfAction, ParseDocxAction, ParseLatexAction, ParseAudioAction,
    ParseImageAction, ParseVideoAction, ParsePptxAction
)

from research_gym.action.empty import SleepAction

from research_gym.action.search import SearchAction

from research_gym.action.browse import (
    WebPageGotoAction, WebPageGotoLineAction, WebPageScrollDownAction,
    WebPageScrollUpAction, WebPageSearchAction, WebPageSearchNextAction,
    WebPageGetLinksAction
)

from research_gym.observation.observation import (
    BaseObservation, CommandObservation, FileObservation, ParseObservation, 
    SearchObservation, WebBrowseObservation,
    ObservationFactory, ObservationType
)

from research_gym.action import BaseAction
from research_gym.action.system import EvalAction
from pathlib import Path
import importlib
from research_gym.configs.task_config import TaskConfig

from evaluations.base.base_eval import BaseBenchmark
from evaluations.base.data_classes import Config

# Define operation handler protocol
@runtime_checkable
class ActionHandler(Protocol):
    """Action handler protocol, defines the interface for handling Actions"""
    
    def can_handle(self, action: BaseAction) -> bool:
        """Check if the given action can be handled"""
        ...
    
    def handle(self, action: BaseAction) -> BaseObservation:
        """Handle action and return observation result"""
        ...

# Abstract base class, provides common implementation
class AbstractActionHandler(ABC):
    """Abstract action handler base class"""
    
    @abstractmethod
    def can_handle(self, action: BaseAction) -> bool:
        """Check if the given action can be handled"""
        pass
    
    @abstractmethod
    def handle(self, action: BaseAction) -> BaseObservation:
        """Handle action and return observation result"""
        pass
    
    def _normalize_result(self, result: Union[Dict[str, Any], str, bool], action: BaseAction) -> Dict[str, Any]:
        """Standardize non-Dict return values to Dict format"""
        if isinstance(result, dict):
            return result
        elif isinstance(result, str):
            # Define success identifiers for file reading operations
            file_reading_prefixes = [
                '[Reading PDF file from',
                '[Reading DOCX file from',
                '[Reading LaTex file from',
                '[Reading audio file from',
                '[Reading image file from',
                '[Reading video file from',
                '[Reading PowerPoint file from'
            ]
            
            if type(action) == ParsePdfAction or type(action) == ParseDocxAction or type(action) == ParseLatexAction or type(action) == ParseAudioAction or type(action) == ParseImageAction or type(action) == ParseVideoAction or type(action) == ParsePptxAction:
                # Check if it is a successful result of file reading operation
                if any(result.startswith(prefix) for prefix in file_reading_prefixes):
                    # TODO: Need to determine success or failure based on result, result may indicate value rather than execution success
                    return {
                        'success': True,
                        'content': result,
                    }
                else:
                    return {
                        'success': False,
                        'error': result,
                    }

            if type(action) == CreateSessionAction:
                # If result is 8-bit string, then create_session is successful
                if len(result) == 8:
                    return {
                        'success': True,
                        'output': result,
                    }
                else:
                    return {
                        'success': False,
                        'error': result,
                    }

            if type(action) == GetSessionRecentOutputAction:
                return {
                    'success': True,
                    'output': result,
                }
            
            return {
                'success': not result.startswith('Error:'),
                'content': result,
                'output': result,
            }


        elif isinstance(result, bool):
            if type(action) == SessionStatusAction:
                return {
                    'success': True,
                    'output': "Session is alive" if result else "Session is not alive",
                }
            
            if type(action) == SessionIdleAction:
                return {
                    'success': True,
                    'output': "Session is idle" if result else "Session is not idle",
                }
            
            return {
                'success': result,
                'result': result,
                'message': f'Operation {"success" if result else "failed"}',
            }
        else:
            # Other types converted to string
            return {
                'success': True,
                'content': str(result),
                'output': str(result),
            }
    
    def _create_error_observation(self, message: str, observation_type: ObservationType = ObservationType.BASE, action: BaseAction = NullAction()) -> BaseObservation:
        """Create error observation result"""
        error_result = {
            'success': False,
            'message': message,
            'output': {'output': []},
        }
        return ObservationFactory.create_observation_by_type(observation_type, error_result, action)

# Generic type, used for type hinting
T = TypeVar('T', bound=BaseAction)

class TypedActionHandler(AbstractActionHandler):
    """Base class for operation handler based on type"""
    
    def __init__(self, action_types: tuple):
        self.action_types = action_types
    
    def can_handle(self, action: BaseAction) -> bool:
        """Check if the action can be handled based on type"""
        return isinstance(action, self.action_types)


class CommandActionHandler(TypedActionHandler):
    """Command operation handler"""
    
    def __init__(self, cmd_operations):
        super().__init__(action_types=(
            CreateSessionAction, ListSessionsAction, RunCommandAction,
            InputInSessionAction, GetSessionOutputAction, GetSessionRecentOutputAction,
            SessionStatusAction, SessionIdleAction, ClearSessionBufferAction,
            CloseSessionAction, CloseAllSessionsAction, KillSessionProcessesAction
        ))
        self.cmd_operations = cmd_operations
    
    def handle(self, action: BaseAction) -> CommandObservation:
        """Handle command related actions, return CommandObservation"""
        try:
            # Command operation mapping dictionary
            operation_map = {
                CreateSessionAction: lambda a: self.cmd_operations.create_session(
                    a.computer_ip, a.session_id, a.http_port, a.use_proxy),
                ListSessionsAction: lambda a: self.cmd_operations.list_sessions(a.computer_ip),
                RunCommandAction: lambda a: self.cmd_operations.run_command(
                    a.command, a.computer_ip, a.session_id, a.http_port, a.wait_for_completion, a.use_proxy),
                InputInSessionAction: lambda a: self.cmd_operations.input_in_session(
                    a.computer_ip, a.session_id, a.input_text),
                GetSessionOutputAction: lambda a: self.cmd_operations.get_session_output(
                    a.computer_ip, a.session_id, a.start_lines, a.end_lines, a.since_timestamp),
                GetSessionRecentOutputAction: lambda a: self.cmd_operations.get_session_recent_output(
                    a.computer_ip, a.session_id, a.seconds),
                SessionStatusAction: lambda a: self.cmd_operations.session_status(
                    a.computer_ip, a.session_id),
                SessionIdleAction: lambda a: self.cmd_operations.session_idle(
                    a.computer_ip, a.session_id),
                ClearSessionBufferAction: lambda a: self.cmd_operations.clear_session_buffer(
                    a.computer_ip, a.session_id),
                CloseSessionAction: lambda a: self.cmd_operations.close_session(
                    a.computer_ip, a.session_id),
                CloseAllSessionsAction: lambda a: self.cmd_operations.close_all_sessions(
                    a.computer_ip),
                KillSessionProcessesAction: lambda a: self.cmd_operations.kill_session_processes(
                    a.computer_ip, a.session_id, a.force),
            }
            
            operation = operation_map.get(type(action))
            if operation:
                result = operation(action)
                # Standardize result to Dict format
                normalized_result = self._normalize_result(result, action)
                # Convert result to CommandObservation
                return ObservationFactory.create_command_observation(normalized_result, action)
            else:
                return self._create_error_observation(
                    f'Unsupported command action: {type(action).__name__}', ObservationType.COMMAND, action)
                
        except Exception as e:
            return self._create_error_observation(
                f'Error handling command action: {str(e)}', ObservationType.COMMAND, action)


class FileActionHandler(TypedActionHandler):
    """File operation handler"""
    
    def __init__(self, file_operations):
        super().__init__(action_types=(
            FileEditAction, OpenFileAction, GotoLineAction, FileScrollDownAction, 
            FileScrollUpAction, CreateFileAction, SearchDirAction, SearchFileAction, 
            FindFileAction, ListFilesAction, GetFileInfoAction
        ))
        self.file_operations = file_operations
    
    def handle(self, action: BaseAction) -> FileObservation:
        """Handle file related actions, return FileObservation"""
        try:
            # File operation mapping dictionary
            operation_map = {
                FileEditAction: lambda a: self.file_operations.edit_file(
                    a.path, a.start, a.end, a.content),
                OpenFileAction: lambda a: self.file_operations.open_file(
                    a.path, a.line_number, a.context_lines),
                GotoLineAction: lambda a: self.file_operations.goto_line(a.line_number),
                FileScrollDownAction: lambda a: self.file_operations.scroll_down(),
                FileScrollUpAction: lambda a: self.file_operations.scroll_up(),
                CreateFileAction: lambda a: self.file_operations.create_file(a.filename, a.content),
                SearchDirAction: lambda a: self.file_operations.search_dir(a.search_term, a.dir_path),
                SearchFileAction: lambda a: self.file_operations.search_file(a.search_term, a.file_path),
                FindFileAction: lambda a: self.file_operations.find_file(a.file_name, a.dir_path),
                ListFilesAction: lambda a: self.file_operations.list_files(a.path, a.show_hidden),
                GetFileInfoAction: lambda a: self.file_operations.get_file_info(),
            }
            
            operation = operation_map.get(type(action))
            if operation:
                result = operation(action)
                # Standardize result to Dict format
                normalized_result = self._normalize_result(result, action)
                # Convert result to FileObservation
                return ObservationFactory.create_file_observation(normalized_result, action)
            else:
                return self._create_error_observation(
                    f'Unsupported file action: {type(action).__name__}', ObservationType.FILE, action)
                
        except Exception as e:
            return self._create_error_observation(
                f'Error handling file action: {str(e)}', ObservationType.FILE, action)


class ParseActionHandler(TypedActionHandler):
    """Parse operation handler"""
    
    def __init__(self, parser_operations):
        super().__init__(action_types=(
            ParsePdfAction, ParseDocxAction, ParseLatexAction, ParseAudioAction,
            ParseImageAction, ParseVideoAction, ParsePptxAction
        ))
        self.parser_operations = parser_operations
    
    def handle(self, action: BaseAction) -> ParseObservation:
        """Handle parse related actions, return ParseObservation"""
        try:
            # Parse operation mapping dictionary
            operation_map = {
                ParsePdfAction: lambda a: self.parser_operations.parse_pdf(a.file_path, a.save_path),
                ParseDocxAction: lambda a: self.parser_operations.parse_docx(a.file_path, a.save_path),
                ParseLatexAction: lambda a: self.parser_operations.parse_latex(a.file_path, a.save_path),
                ParseAudioAction: lambda a: self.parser_operations.parse_audio(a.file_path, a.save_path, a.model),
                ParseImageAction: lambda a: self.parser_operations.parse_image(a.file_path, a.save_path, a.task),
                ParseVideoAction: lambda a: self.parser_operations.parse_video(
                    a.file_path, a.save_path, a.task, a.frame_interval),
                ParsePptxAction: lambda a: self.parser_operations.parse_pptx(a.file_path, a.save_path),
            }
            
            operation = operation_map.get(type(action))
            if operation:
                result = operation(action)
                # Standardize result to Dict format
                normalized_result = self._normalize_result(result, action)
                # Convert result to ParseObservation
                return ObservationFactory.create_parse_observation(normalized_result, action)
            else:
                return self._create_error_observation(
                    f'Unsupported parse action: {type(action).__name__}', ObservationType.PARSE, action)
                
        except Exception as e:
            return self._create_error_observation(
                f'Error handling parse action: {str(e)}', ObservationType.PARSE, action)


class ThinkingActionHandler(TypedActionHandler):
    """Thinking and system action handler"""
    
    def __init__(self):
        super().__init__(action_types=(ThinkAction))
    
    def handle(self, action: BaseAction) -> BaseObservation:
        """Handle thinking and system actions, return base observation result"""
        try:
            if isinstance(action, ThinkAction):
                result = {
                    'success': True,
                    'content': action.thought,
                    'output': action.thought,
                    'message': action.thought
                }
                return ObservationFactory.create_observation_by_type(ObservationType.BASE, result, action)
            else:
                return self._create_error_observation(
                    f'Unsupported action: {type(action).__name__}', ObservationType.BASE, action)
                
        except Exception as e:
            return self._create_error_observation(
                f'Error handling action: {str(e)}', ObservationType.BASE, action) 


class ViewHintActionHandler(TypedActionHandler):
    def __init__(self, task_config: TaskConfig):
        super().__init__(action_types=(ViewHintAction,))
        self.task_config = task_config

    def handle(self, action: BaseAction) -> BaseObservation:
        """Handle check hint action, return base observation result"""
        try:
            hint_file_path = os.path.join(self.task_config.eval_workspace, self.task_config.task_name, "hint.md")
            if os.path.exists(hint_file_path):
                with open(hint_file_path, 'r', encoding='utf-8') as f:
                    hint_content = f.read()
                
                # stat_result = os.stat(hint_file_path)
                
                # lines = hint_content.split('\n')
                
                result = {
                    'success': True,
                    'message': hint_content,
                    'error': ""
                }
            else:
                result = {
                    'success': False,
                    'file': hint_file_path,
                    'content': "",
                    'file_exists': False,
                    'error': "hint not found"
                }
            return ObservationFactory.create_file_observation(result, action)
        except Exception as e:
            return self._create_error_observation(
                f'Error handling view hint action: {str(e)}', ObservationType.FILE, action)


class EvalActionHandler(TypedActionHandler):
    """Eval operation handler"""
    
    def __init__(self, task_config: TaskConfig):
        super().__init__(action_types=(EvalAction,))
        self.task_config = task_config
        # TODO: Monitor maximum number of runs
        self.max_eval_num = task_config.max_eval_num
        self.eval_num = 0
        
    def load_eval_class(self, path: Path, class_type: str) -> BaseBenchmark | Config:
        """
        path should point to evaluations/<task_name>/task/evaluation.py
        """
        task_name = self.task_config.task_name  # For example "task_1"

        if not path.exists():
            raise FileNotFoundError(f"Evaluation {class_type} for task {task_name} not found: {path}")

        # 1) Insert the "project root directory" (the layer containing evaluations/) into sys.path
        # path = .../evaluations/task_1/task/evaluation.py
        # parents[0]=.../task, [1]=.../task_1, [2]=.../evaluations, [3]=.../ai-engineer-benchmark ← need this
        project_root = str(path.resolve().parents[3])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # 2) Import evaluation module by package name (to make relative imports work)
        
        if class_type.lower() in ("benchmark", "eval", "evaluation"):
            module_name = f"evaluations.{task_name}.task.evaluation"
        elif class_type.lower() in ("config"):
            module_name = f"evaluations.{task_name}.task.config"
        else:
            raise ValueError("load_eval_class error")
        try:
            eval_module = import_module(module_name)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Cannot import module {module_name}. Please ensure that both evaluations/ and {task_name}/ directories have __init__.py files, "
                f"and that sys.path includes the project root directory: {project_root}. Original error: {e}"
            )

        # 3) Get class name by convention
        # Your evaluation.py defines TaskBenchmark
        if class_type.lower() in ("benchmark", "eval", "evaluation"):
            class_name = "TaskBenchmark"
        elif class_type.lower() in ("config"):
            class_name = "TaskConfig"
        else:
            raise ValueError("load_eval_class error")

        try:
            return getattr(eval_module, class_name)
        except AttributeError:
            raise AttributeError(f"Cannot find class {class_name} in module {module_name}")
        
    def handle(self, action: BaseAction) -> BaseObservation:
        """Handle evaluation action, return base observation result"""
        self.eval_num += 1
        if self.eval_num > self.max_eval_num and action.call_id != "00000002":
            result = {
                "task_name": self.task_config.task_name,
                "score": 0,
                "eval_results": {},
                "already_eval_num": self.eval_num,
                "success": True,
                "message": "Eval finish!",
            }
            return ObservationFactory.create_observation_by_type(ObservationType.EVAL, result, action)
        
        
        if not isinstance(action, EvalAction):
            return self._create_error_observation("Invalid action type for EvalActionHandler", ObservationType.BASE, action)

        try:
            task_name = self.task_config.task_name

            eval_script_path = Path(os.path.join(self.task_config.eval_workspace, task_name, "task", "evaluation.py"))
            eval_config_path = Path(os.path.join(self.task_config.eval_workspace, task_name, "task", "config.py"))

            # Dynamically get the benchmark class, e.g., TaskBenchmark
            eval_class: BaseBenchmark = self.load_eval_class(eval_script_path, "Benchmark")
            eval_config_class: Config = self.load_eval_class(eval_config_path, "Config")

            config = eval_config_class(
                eval_workspace=self.task_config.eval_workspace,
                workspace=self.task_config.workspace,
            )
            # Instantiate the benchmark and validate
            benchmark = eval_class(config)
            eval_results = benchmark.validate()

            print("eval results: ", eval_results)

            if self.max_eval_num - self.eval_num > 0:
                
                # "eval_results": eval_results,
                result = {
                    "task_name": self.task_config.task_name,
                    "eval_results": {},
                    "already_eval_num": self.eval_num,
                    "success": True,
                    "message": f"Eval finish! You have remaining {self.max_eval_num - self.eval_num} eval times.", 
                }
            else:
                # "eval_results": eval_results,
                result = {
                    "task_name": self.task_config.task_name,
                    "eval_results": {},
                    "already_eval_num": self.eval_num,
                    "success": True,
                    "message": f"Eval finish! You have remaining 0 eval times. If you use eval in the future, the task will automatically finish.", 
                }
            score = 0
            for i, eval_result in enumerate(eval_results):
                eval_result = eval_result.to_dict()
                print("eval_result: ", eval_result)
                assert "score" in eval_result["eval_results"], f"score must be a key in result of {benchmark}." 
                score += eval_result["eval_results"]["score"]
                del eval_result["task_name"]
                result["eval_results"][f"metric_{i}"] = eval_result
            result["score"] = min(score, 100)
            return ObservationFactory.create_observation_by_type(ObservationType.EVAL, result, action)
        except Exception as e:
            return self._create_error_observation(
                f'Error handling eval action: {str(e)}', ObservationType.EVAL, action)


class SleepActionHandler(TypedActionHandler):
    """Sleep action handler"""
    
    def __init__(self):
        super().__init__(action_types=(SleepAction,))
    
    def handle(self, action: BaseAction) -> BaseObservation:
        """Handle sleep action, return base observation result"""
        try:
            if isinstance(action, SleepAction):
                # Validate sleep time
                sleep_time = max(1e-6, min(action.sleep_time, 1800))  # Maximum 1800 seconds

                print(f"Sleep for {sleep_time/60} minutes")
                
                # Execute sleep
                from time import sleep
                sleep(sleep_time)
                
                result = {
                    'success': True,
                    'content': f"Slept for {sleep_time} seconds",
                    'output': f"Sleep completed: {sleep_time}s",
                    'message': f"Sleep completed. Please check the situation of the environment first (e.g. Is the command still alive in the terminal?). Do not sleep again in the next action. But you can sleep in the future."
                }
                return ObservationFactory.create_observation_by_type(ObservationType.BASE, result, action)
            else:
                return self._create_error_observation(
                    f'Unsupported sleep action: {type(action).__name__}', ObservationType.BASE, action)

        except Exception as e:
            return self._create_error_observation(
                f'Error handling sleep action: {str(e)}', ObservationType.BASE, action)


class SearchActionHandler(TypedActionHandler):
    """Search action handler"""
    
    def __init__(self, search_operations):
        super().__init__(action_types=(SearchAction,))
        self.search_operations = search_operations
    
    def handle(self, action: BaseAction) -> BaseObservation:
        """Handle search action, return base observation result"""
        try:
            if isinstance(action, SearchAction):
                # Validate top_k parameter
                top_k = max(1, min(action.top_k, 100))  # Limit to 1-100 range

                # Execute search
                result = self.search_operations.search(action.query, top_k)

                # Standardize result
                if result.get('success'):
                    # Successful search results - fix key mismatch issue and add all necessary fields
                    search_results = result.get('search_results', [])
                    normalized_result = {
                        'success': True,
                        'content': f"Found {len(search_results)} search results for '{action.query}'",
                        'message': result.get('message', f"Found {len(search_results)} search results"),
                        'output': result,
                        'search_results': search_results,
                        'query': action.query,
                        'requested_top_k': result.get('requested_top_k', action.top_k),
                        'total_results': result.get('total_results', len(search_results)),
                        'from_cache': result.get('from_cache', False)
                    }
                else:
                    # Search failed - fix error message key mismatch issue
                    error_message = result.get('message', 'Unknown error')
                    normalized_result = {
                        'success': False,
                        'content': f"Search failed: {error_message}",
                        'message': error_message,
                        'error_code': result.get('error_code'),
                        'query': action.query
                    }
                
                return ObservationFactory.create_observation_by_type(ObservationType.WEB_SEARCH, normalized_result, action)
            else:
                return self._create_error_observation(
                    f'Unsupported search action: {type(action).__name__}', ObservationType.WEB_SEARCH, action)

        except Exception as e:
            return self._create_error_observation(
                f'Error handling search action: {str(e)}', ObservationType.WEB_SEARCH, action)


class WebActionHandler(TypedActionHandler):
    """Web browsing action handler"""
    
    def __init__(self, web_operations):
        super().__init__(action_types=(
            WebPageGotoAction, WebPageGotoLineAction, WebPageScrollDownAction,
            WebPageScrollUpAction, WebPageSearchAction, WebPageSearchNextAction,
            WebPageGetLinksAction
        ))
        self.web_operations = web_operations
    
    def handle(self, action: BaseAction) -> BaseObservation:
        """Handle web browsing action, return base observation result"""
        try:
            # Web operation mapping dictionary
            operation_map = {
                WebPageGotoAction: lambda a: self.web_operations.goto(a.url, a.line_number),
                WebPageGotoLineAction: lambda a: self.web_operations.goto_line(a.line_number),
                WebPageScrollDownAction: lambda a: self.web_operations.scroll_down(),
                WebPageScrollUpAction: lambda a: self.web_operations.scroll_up(),
                WebPageSearchAction: lambda a: self.web_operations.search(a.keyword, a.context_lines),
                WebPageSearchNextAction: lambda a: self.web_operations.search_next(
                    a.context_lines, a.search_index),
                WebPageGetLinksAction: lambda a: self.web_operations.get_links(
                    a.page_size, a.page_number),
            }
            
            operation = operation_map.get(type(action))
            if operation:
                result = operation(action)

                # Standardize result to Dict format (web_operations already returns dict format)
                if isinstance(result, dict):
                    # If already dict, use directly
                    normalized_result = result
                else:
                    # If other format, standardize
                    normalized_result = self._normalize_result(result, action)
                
                # Convert result to WebBrowseObservation
                return ObservationFactory.create_observation_by_type(ObservationType.WEB_BROWSE, normalized_result, action)
            else:
                return self._create_error_observation(
                    f'Unsupported web action: {type(action).__name__}', ObservationType.WEB_BROWSE, action)
                
        except Exception as e:
            return self._create_error_observation(
                f'Error handling web action: {str(e)}', ObservationType.WEB_BROWSE, action)