from dataclasses import dataclass, field
from typing import ClassVar, Optional, Dict, Any, List

from research_gym.action.action import BaseAction, ActionConfirmationStatus, ActionSecurityRisk, ToolCallArguments
from research_gym.schema.action import ActionType


@dataclass
class CommandBaseAction(BaseAction):
    """Command base action"""
    computer_ip: str = field(
        default=None,
        metadata={
            'tool_param': True,
            'description': 'The IP address of the computer.',
            'required': True
        }
    )


@dataclass
class CreateSessionAction(CommandBaseAction):
    """Create new terminal session"""
    session_id: str = field(
        default='',
        metadata={
            'tool_param': True,
            'description': 'Unique identifier of the target session. If absent, a new session is created and a new `session_id` is assigned on the host `computer_ip`.',
            'required': True
        }
    )
    http_port: Optional[int] = field(
        default=None,
        metadata={
            'tool_param': True,
            'description': 'The HTTP port to use to connnect to the session.',
            'required': True
        }
    )
    use_proxy: bool = field(
        default=True,
        metadata={
            'tool_param': True,
            'description': 'Whether to use a proxy for connecting to the session. Set `use_proxy=False` for `cpu` and `localhost_cpu` computers, and set `use_proxy=True` for `gpu` computers. Must align with your network topology or the connection will fail.',
            'required': True
        }
    )
    action_type: str = ActionType.CREATE_SESSION
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.LOW
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    description: str = "Create a new terminal session on the computer specified by `computer_ip`, initializing connectivity via `http_port` and `use_proxy`. Use `use_proxy=False` for `cpu`/`localhost_cpu` machines and `use_proxy=True` for `gpu` machines."

    @property
    def message(self) -> str:
        target = self.container_name if self.container_name else self.computer_ip
        return f'Creating session on {target}' + (f' with ID {self.session_id}' if self.session_id else '')

    def __str__(self) -> str:
        ret = f'**CreateSessionAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        if self.computer_ip:
            ret += f'CPU IP: {self.computer_ip}\n'
        if self.session_id:
            ret += f'Session ID: {self.session_id}\n'
        if self.http_port:
            ret += f'HTTP Port: {self.http_port}\n'
        ret += f'Use Proxy: {self.use_proxy}'
        return ret


@dataclass
class ListSessionsAction(CommandBaseAction):
    """List active sessions"""
    
    action_type: str = ActionType.LIST_SESSIONS
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.LOW
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    description: str = "List all exist sessions. Key '<computer_ip>:<session_id>' on the output refers to the session <session_id> on <computer_ip>"

    @property
    def message(self) -> str:
        return f'Listing sessions' + (f' on {self.computer_ip}' if self.computer_ip else ' on all machines')

    def __str__(self) -> str:
        ret = f'**ListSessionsAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'Target: {self.computer_ip or "all machines"}'
        return ret


@dataclass
class RunCommandAction(CommandBaseAction):
    """Execute command in specified session (enhanced CmdRunAction)"""
    
    # Tool parameters: use metadata marking
    command: str = field(
        default='',
        metadata={
            'tool_param': True,
            'description': 'Shell (bash) command to execute in the target session\'s working directory and environment. You can\'t use `kill`, `pkill`, and `killall` command. You can\'t use `rm` outside the `/workspace`',
            'required': True
        }
    )
    session_id: str = field(
        default=None,
        metadata={
            'tool_param': True,
            'description': 'Unique identifier of the target session. If absent, a new session is created on the host determined by `computer_ip`. Only one command may run concurrently per session; concurrent invocations will be rejected.',
            'required': True
        }
    )
    http_port: int = field(
        default=None,
        metadata={
            'tool_param': True,
            'description': 'The HTTP port to use to connnect to the session.',
            'required': True
        }
    )
    wait_for_completion: bool = field(
        default=False,
        metadata={
            'tool_param': True,
            'description': 'Whether to block until the command finishes:\n - True: block up to 10 seconds; on timeout the command process is killed. Best for short commands (e.g. `ps -aux`).\n - False: return immediately and let the command run in the background. Later use `get_session_output` to view results (e.g. reading data, training, and inference) or use `check_session_idle` to check if the process finish (if not this actiob will tell you which subprocess is running).',
            'required': True
        }
    )
    use_proxy: bool = field(
        default=True,
        metadata={
            'tool_param': True,
            'description': 'Whether to use a proxy for connecting to the session. Set `use_proxy=False` for `cpu` and `localhost_cpu` computers, and set `use_proxy=True` for `gpu` computers. Must align with your network topology or the connection will fail.',
            'required': True
        }
    )
    container_name: Optional[str] = None
    action_type: str = ActionType.RUN
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.MEDIUM
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    description: str = "Execute a single bash command in the session identified by `session_id`. If the session does not exist, it will be created and bound to the target host (determined by `computer_ip`) and will be connected via `http_port` and `use_proxy`. Only one command may run concurrently per session. Set `use_proxy=False` for `cpu` and `localhost_cpu` computers, and set `use_proxy=True` for `gpu` computers. With `wait_for_completion=True`, the command can be run up to 10 seconds; if the command has not finished, the process is killed. For long-running jobs (IO/training/inference), set `wait_for_completion=False` and view results via `get_session_output` tool or `check_session_idle` tool."

    @property
    def message(self) -> str:
        target = self.container_name if self.container_name else self.computer_ip
        return f'Running command on {target}: {self.command}'

    def __str__(self) -> str:
        ret = f'**RunCommandAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        # Prioritize displaying container name
        if self.computer_ip:
            ret += f'CPU IP: {self.computer_ip}\n'
        if self.session_id:
            ret += f'Session ID: {self.session_id}\n'
        ret += f'Wait for completion: {self.wait_for_completion}\n'
        ret += f'COMMAND:\n{self.command}'
        return ret


@dataclass
class InputInSessionAction(CommandBaseAction):
    """Send input to specified session"""
    session_id: str = field(
        default='',
        metadata={
            'tool_param': True,
            'description': 'Unique identifier of the target session. Must refer to an existing, active session. If the session has no foreground command awaiting stdin, the call will be rejected.',
            'required': True
        }
    )
    input_text: str = field(
        default='',
        metadata={
            'tool_param': True,
            'description': 'Text to write to the process\'s stdin.',
            'required': True
        }
    )

    action_type: str = ActionType.INPUT_IN_SESSION
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.LOW
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    description: str = "Send stdin text to a specific terminal session, valid only when its foreground command is awaiting input. If the session is not awaiting input, the request is rejected."

    @property
    def message(self) -> str:
        return f'Sending input to session on {self.computer_ip}: {self.input_text[:50]}...'

    def __str__(self) -> str:
        ret = f'**InputInSessionAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'CPU IP: {self.computer_ip}\n'
        if self.session_id:
            ret += f'Session ID: {self.session_id}\n'
        ret += f'INPUT:\n{self.input_text}'
        return ret


@dataclass
class GetSessionOutputAction(CommandBaseAction):
    """Get session output"""
    session_id: str = field(
        default='',
        metadata={
            'tool_param': True,
            'description': 'Unique identifier of the target session. The session must exist and be active.',
            'required': True
        }
    )
    start_lines: int = field(
        default=50,
        metadata={
            'tool_param': True,
            'description': 'Start offset counted from the **end of output** (>=2). Effective only when `since_timestamp` is not set.\nUsage:\n- `start_lines=N` only: returns the **last N lines**.\n- With `end_lines`: returns the slice between `start_lines` and `end_lines`.',
            'required': True
        }
    )
    end_lines: Optional[int] = field(
        default=None,
        metadata={
            'tool_param': True,
            'description': 'End offset counted from the **end of output** (>=1). If not specified, this tool will return content from the `start_lines` to the end of the output. If specified, the slice is **[start_lines, end_lines)**: inclusive of `start_lines`, exclusive of `end_lines`.\nExample: `start_lines=100, end_lines=50` returns lines between the 100th-from-end and 50th-from-end.',
            'required': False
        }
    )
    since_timestamp: float = field(
        default=None,
        metadata={
            'tool_param': True,
            'description': 'Optional. Fetch output **since** this Unix epoch timestamp (seconds, float). When set, it **overrides** `start_lines` and `end_lines`. Useful for polling: store the last-read timestamp and pass it on next call.',
            'required': False
        }
    )
    action_type: str = ActionType.GET_SESSION_OUTPUT
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.LOW
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    description: str = "Retrieve the output buffer of the terminal session identified by `session_id` (on the host specified by `computer_ip`). If `since_timestamp` is provided, incremental output since that time is returned; otherwise, output is sliced by line window (`start_lines` required, `end_lines` optional). If the buffer exceeds 30,000 characters, only the newest 30,000 characters are returned. ## Remember: this action will return all available outputs in the session from last `start_lines` to last `end_lines`. It may comtain the log of the previous commands. You need to seperate it via excpetion log or interactive prompt"
    
    @property
    def message(self) -> str:
        target = self.container_name if self.container_name else self.computer_ip
        return f'Getting output from session on {target}'

    def __str__(self) -> str:
        ret = f'**GetSessionOutputAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        if self.computer_ip:
            ret += f'CPU IP: {self.computer_ip}\n'
        if self.session_id:
            ret += f'Session ID: {self.session_id}\n'
        ret += f'Start lines: {self.start_lines}\n'
        if self.end_lines:
            ret += f'End lines: {self.end_lines}\n'
        if self.since_timestamp:
            ret += f'Since timestamp: {self.since_timestamp}'
        return ret


@dataclass
class GetSessionRecentOutputAction(CommandBaseAction):
    """Get session output from recent seconds"""
    
    session_id: str = field(
        default='',
        metadata={
            'tool_param': True,
            'description': 'The ID of the session to get the recent output.',
            'required': True
        }
    )
    seconds: int = field(
        default=10,
        metadata={
            'tool_param': True,
            'description': 'The number of seconds to get the recent output from.',
            'required': True
        }
    )
    action_type: str = ActionType.GET_SESSION_RECENT_OUTPUT
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.LOW
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    description: str = "Get the recent output from a specific terminal session output buffer."

    @property
    def message(self) -> str:
        return f'Getting recent output ({self.seconds}s) from session on {self.computer_ip}'

    def __str__(self) -> str:
        ret = f'**GetSessionRecentOutputAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'CPU IP: {self.computer_ip}\n'
        if self.session_id:
            ret += f'Session ID: {self.session_id}\n'
        ret += f'Seconds: {self.seconds}'
        return ret


@dataclass
class SessionStatusAction(CommandBaseAction):
    """Get session status"""
    
    session_id: str = field(
        default='',
        metadata={
            'tool_param': True,
            'description': 'The ID of the session to get its status.',
            'required': True
        }
    )
    # Added: container name parameter
    container_name: Optional[str] = None
    action_type: str = ActionType.GET_SESSION_STATUS
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.LOW
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    description: str = "Get the status of a specific terminal session. You can use this action to check if the session is running a command."

    @property
    def message(self) -> str:
        target = self.container_name if self.container_name else self.computer_ip
        return f'Checking session status on {target}'

    def __str__(self) -> str:
        ret = f'**SessionStatusAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        if self.container_name:
            ret += f'Container: {self.container_name}\n'
        else:
            ret += f'CPU IP: {self.computer_ip}\n'
        if self.session_id:
            ret += f'Session ID: {self.session_id}'
        return ret


@dataclass
class SessionIdleAction(CommandBaseAction):
    """Check if session is idle"""
    
    session_id: str = field(
        default='',
        metadata={
            'tool_param': True,
            'description': 'The ID of the session to check whether it is running some command or whether it is idle.',
            'required': True
        }
    )
    action_type: str = ActionType.CHECK_SESSION_IDLE
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.LOW
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    description: str = "Check if a specific terminal session is idle"

    @property
    def message(self) -> str:
        return f'Checking if session is idle on {self.computer_ip}'

    def __str__(self) -> str:
        ret = f'**SessionIdleAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'CPU IP: {self.computer_ip}\n'
        if self.session_id:
            ret += f'Session ID: {self.session_id}'
        return ret


@dataclass
class ClearSessionBufferAction(CommandBaseAction):
    """Clear session output buffer"""
    
    session_id: str = field(
        default='',
        metadata={
            'tool_param': True,
            'description': 'The ID of the session to clear the output buffer.',
            'required': True
        }
    )
    action_type: str = ActionType.CLEAR_SESSION_BUFFER
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.LOW
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    description: str = "Clear the output buffer of a specific terminal session. (The output buffer is a queue of output lines, it will automatically clean if the total lines exceed 10000 lines, regardless of using this action or not.)"

    @property
    def message(self) -> str:
        return f'Clearing session buffer on {self.computer_ip}'

    def __str__(self) -> str:
        ret = f'**ClearSessionBufferAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'CPU IP: {self.computer_ip}\n'
        if self.session_id:
            ret += f'Session ID: {self.session_id}'
        return ret


@dataclass
class CloseSessionAction(CommandBaseAction):
    """Close specified session"""
    
    session_id: str = field(
        default='',
        metadata={
            'tool_param': True,
            'description': 'The ID of the session to close.',
            'required': True
        }
    )
    action_type: str = ActionType.CLOSE_SESSION
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.MEDIUM
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    description: str = "Close a specific terminal session and kill all sub-processes in the session."

    @property
    def message(self) -> str:
        return f'Closing session {self.session_id} on {self.computer_ip}'

    def __str__(self) -> str:
        ret = f'**CloseSessionAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'CPU IP: {self.computer_ip}\n'
        ret += f'Session ID: {self.session_id}'
        return ret


@dataclass
class CloseAllSessionsAction(CommandBaseAction):
    """Close all or specified machine's all sessions"""
    
    action_type: str = ActionType.CLOSE_ALL_SESSIONS
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.HIGH
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.AWAITING_CONFIRMATION
    description: str = "Close all sessions on a specific machine or all machines. If you want to close all sessions on a specific machine, you should set the `computer_ip`."

    @property
    def message(self) -> str:
        return f'Closing all sessions' + (f' on {self.computer_ip}' if self.computer_ip else ' on all machines')

    def __str__(self) -> str:
        ret = f'**CloseAllSessionsAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'Target: {self.computer_ip or "all machines"}'
        return ret


@dataclass
class KillSessionProcessesAction(CommandBaseAction):
    """Kill all subprocesses in the session"""
    
    session_id: str = field(
        default='',
        metadata={
            'tool_param': True,
            'description': 'The ID of the session to kill all processes.',
            'required': True
        }
    )
    force: bool = field(
        default=False,
        metadata={
            'tool_param': True,
            'description': 'Whether to force to kill all processes.',
            'required': True
        }
    )
    action_type: str = ActionType.KILL_SESSION_PROCESSES
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk = ActionSecurityRisk.HIGH
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.AWAITING_CONFIRMATION
    description: str = "Kill all processes on a specific session"

    @property
    def message(self) -> str:
        return f'Killing processes in session on {self.computer_ip}' + (' (force)' if self.force else '')

    def __str__(self) -> str:
        ret = f'**KillSessionProcessesAction (source={self.source})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'CPU IP: {self.computer_ip}\n'
        if self.session_id:
            ret += f'Session ID: {self.session_id}\n'
        ret += f'Force: {self.force}'
        return ret 