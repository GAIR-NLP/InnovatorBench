from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, List, Any, Dict

from research_gym.action.action import BaseAction, ActionSecurityRisk, ToolCallArguments
from research_gym.schema.action import ActionType

class AgentFinishTaskCompleted(str, Enum):
    FALSE = 'false'
    PARTIAL = 'partial'
    TRUE = 'true'
    FAIL = 'fail'

@dataclass
class FinishAction(BaseAction):
    """An action where the agent finishes the task.

    Attributes:
        final_thought (str): The message to send to the user.
        task_completed (enum): Whether the agent believes the task has been completed.
        outputs (dict): The other outputs of the agent, for instance "content".
        thought (str): The agent's explanation of its actions.
        action (str): The action type, namely ActionType.FINISH.
    """


    outputs: dict[str, Any] = field(default_factory=dict)
    action_type: ClassVar[str] = ActionType.FINISH
    description: str = "Finish the original task. If you believe the task is completed (saved all the files that the original task requires and the result can't be improved anymore), you can use this action. You can only finish the task once. The argument of this action should be empty, do not add any key inside the argument"
    task_completed: AgentFinishTaskCompleted = AgentFinishTaskCompleted.TRUE

    @property
    def message(self) -> str:
        if self.thought != '':
            return self.thought
        if self.task_completed == AgentFinishTaskCompleted.TRUE:
            return "All done! What's next on the agenda?"
        elif self.task_completed == AgentFinishTaskCompleted.FALSE:
            return "I'm not sure if I've completed the task. Please check my work."
        elif self.task_completed == AgentFinishTaskCompleted.PARTIAL:
            return "I've completed part of the task. Please check my work and continue."
        elif self.task_completed == AgentFinishTaskCompleted.FAIL:
            return "I've failed to complete the task. Please check my work."
        else:
            self.task_completed = AgentFinishTaskCompleted.TRUE
            return "All done! What's next on the agenda?"

    def __str__(self) -> str:
        ret = f'**FinishAction**\n'
        if self.thought:
            ret += f'Thought: {self.thought}\n'
        return ret

@dataclass
class ThinkAction(BaseAction):
    """An action where the agent logs a thought.

    Attributes:
        thought (str): The agent's explanation of its actions.
        action_type (str): The action type, namely ActionType.THINK.
    """
    thought: str = field(
        default='',
        metadata={
            'tool_param': True,
            'description': "Your explanation of your actions."
        }
    )
    action_type: str = ActionType.THINK
    description: str = "An action where you log a concise internal thought/rationale for traceability. Explain the decision or next step and why it helps, less than 200 characters."
    @property
    def message(self) -> str:
        return f'I am thinking...: {self.thought}'
    
    def __str__(self) -> str:
        ret = f'**ThinkAction**\n'
        if self.thought:
            ret += f'Thought: {self.thought}\n'
        return ret

@dataclass
class ViewHintAction(BaseAction):
    """An action where the agent views the hint.

    Attributes:
        action_type (str): The action type, namely ActionType.VIEW_HINT.
    """
    action_type: str = ActionType.VIEW_HINT
    description: str = "Some tasks contains hints, you can use this action to view the hint, but using this action will deduct your score. Suggest using this action if you got score less than 50."

    @property
    def message(self) -> str:
        return 'I am viewing the hint...'

    def __str__(self) -> str:
        ret = f'**ViewHintAction**\n'
        if self.thought:
            ret += f'Thought: {self.thought}\n'
        return ret

@dataclass
class SummarizeAction(BaseAction):
    """An action where the agent summarizes the current state.

    Attributes:
        action_type (str): The action type, namely ActionType.SUMMARIZE.
    """

    start_summary_depth: int = field(default=1, metadata={
        'tool_param': True,
        'description': 'The depth of the turn the summary to start with (The summarization will include this turn).',
        'required': True
    })
    end_summary_depth: int = field(default=3, metadata={
        'tool_param': True,
        'description': 'The depth of the turn the summary to end with (The summarization will not include this turn).',
        'required': True
    })

    action_type: str = ActionType.SUMMARIZE
    description: str = "An action where the agent summarizes the current state. The depth i context means the context inside [START OF DEPTH i] and [END OF DEPTH i]. Remember: you can only summarize the context after the last task/subtask turn. The last task/subtask is the turn that you try to decompose the task (or the original task if there is no decomposition) and the user accept your decomposition (like \"ok, In the following turns, you need to try to finish the subgoal <subgoal>\" )."

    @property
    def message(self) -> str:
        return f'I am summarizing...: {self.thought}'

    def __str__(self) -> str:
        ret = f'**SummarizeAction**\n'
        if self.thought:
            ret += f'Thought: {self.thought}\n'
        if self.start_summary_depth:
            ret += f'Start summary depth: {self.start_summary_depth}\n'
        if self.end_summary_depth:
            ret += f'End summary depth: {self.end_summary_depth}\n'
        return ret


@dataclass
class InternalSummarizeAction(BaseAction):

    action_type: str = ActionType.INTERNAL_SUMMARIZE
    description: str = "An action where the agent summarizes the current state. This action is used for internal use, you can only use this action when you are in the summary node."
    summary_content: str = field(default="", metadata={
        'tool_param': True,
        'description': """# The structure of `summary_content` MUST be as follows:

<state_snapshot>
    <state_of_the_art>
        <!-- The SOTA benchmark to surpass. -->
        <!-- Example: "The current SOTA score is 0.85. We need to beat this." -->
    </state_of_the_art>

    <hypotheses>
        <!-- List of active, tested, or pending hypotheses. -->
        <!-- Example:
         - [TESTING] Hypothesis 1: Adding a penalty for verbosity in the reward function will improve conciseness without harming helpfulness.
         - [PROVEN] Hypothesis 2: Normalizing rewards by batch statistics stabilizes training.
         - [TODO] Hypothesis 3: Using data augmentation on the prompt dataset will increase instruction-following capabilities.
        -->
    </hypotheses>

    <key_knowledge>
        <!-- Crucial facts, takeaways, and constraints the agent must remember based on the conversation history and interaction with the user. Use bullet points. -->
        <!-- Example:
         - Ray: ray has started with \`ray start --head\` but havn't check its status.
         - API Endpoint: The primary API endpoint is \`https://api.example.com/v2\`.
         - Learning rate > 1e-4 causes training instability.
         - The main dataset is located at '/data/datasets/rl_dataset_v2.parquet'.
         - Model weights are at '/data/checkpoints/base_model.pth'.
         - Trainging models: llamafactory-cli has been started, the response of the training data is generated by the Qwen2.5-72B-Instruct model.
         - The number of remaining calls to the `eval` tool is 2.
         - Reading File: The `test.parquet` data's value is too long, I should read the special key
        -->
    </key_knowledge>
    
    <reflection> 
        <!-- Reflection that the agent should remember based on conversation history and interactions. Use bullet points. -->
        <!-- Present the reasoning step concisely when stating an Reflection. -->
        <!-- Each line should be in the format of: `Reflection: concise reasoning step and its corresponding facts in the history. -->
        <!-- Only add, edit or merge reflection when there are some incidents in the history. Do not generate redundant reflections. -->
        <!-- The reflections should be general. -->
        <!-- Add reflection from below examples when they appear in the history; you are encouraged to create new, relevant reflection or edit reflection towards new situation. --> 
        <!-- If this reflection comes from real user's advice (content inside <real_user> tag), cite its input in [real_user][/real_user]. -->
        <!-- Examples:
        - Use a special key to read file: in `test.parquet`, some values are very long; reading directly may exceed the context length.
        - Use `wait_for_completion=False` for Ray/training/inference jobs lasting >10 seconds; in the past, jobs were killed when `wait_for_completion=True`.
        - Check GPU status before training/inference: once, training started while another process was already running, causing confusion and wasted time debugging the conda environment. [real_user]Do not running this inference scripts. You have already run another training scripts[/real_user]
        - Always check the file after editing to avoid unexpected modification. 
        - Run commands in the correct path: if not run in folder `A`, Python may import the environment's `math` module instead of `A/math.py`, even with `sys.path.append('A')`.
        - Be patient: importing `transformers` or starting Ray can take about 5 minutes; avoid killing the process prematurely.  [real_user]Your training script is right, why you kill this script?[/real_user]
        - Do not specify `end_lines` in most cases: you often need to read the tail of the session to get the newest information.
        - Determine scope: only the information after the last exception log or interactive prompt is the last command's output; confusion often happens when `start_lines` is set too large.
        - Check the session's status and kill unused sessions after planning/summarization: a run was started and forgotten, leading to duplicate launches; verify idleness and the latest output before starting again.
        -->
    </reflection>

    <file_and_browser_state>
        <!-- List files that have been created, read, modified, deleted and key data artifacts. Note their status and critical learnings. -->
        <!-- Example:
         - CWD: `/workspace/task/`
         - MODIFIED: `/workspace/task/reward.py` - Implemented the verbosity penalty.
         - CREATED: `/workspace/task/scripts/data_augmentation.py` - Script to apply back-translation.
         - DATASET: `/workspace/data/datasets/augmented_prompts.json` - New dataset created from Hypothesis 3.
         - READING: \`README.md\` - The last file you are opening/reading.
         - BROWSED: \`https://www.google.com/search?q=new+feature\` - The last browswe page you have visited.
        -->
    </file_and_browser_state>

    <recent_sessions>
        <!-- List **all** sessions that have been created and not been closed. Note their status and critical learnings. -->
        <!-- Only the session maybe running will have GPU usage. If the running is finish, GPU usage should be None. -->
        <!-- Idle means there is no process running in this session, if one process is end and not run other command in the session, this session is idle -->
        <!-- Highlight the GPU that may have conflict in different session -->
        <!-- Example:
         - [session ID1] Last command: [Command in session ID1], Idle: False, GPU usage: computer ip xxx.xxx.xxx.xxx GPU 0,1,2,3,4,5,6,7 and computer ip xxx.xxx.xxx.xxx GPU 0,1,2,3,4,5,6,7 
         - [session ID2] Last command: [Command in session ID2], Idle: True, GPU usage: None
         - [session ID3] Last command: [Command in session ID3], Idle: False,  GPU usage: computer ip xxx.xxx.xxx.xxx GPU 0,1,2,3
        -->
    </recent_sessions>

    <recent_actions>
        <!-- A summary of the last few significant agent actions and their outcomes. Focus on facts. -->
        <!-- Example:
         - Ran \`grep 'old_function'\` in session xxxxxxxx, computer ip xxx.xxx.xxx.xxx which returned 3 results in 2 files.
         - Ran \`bash inference.sh\` in session xxxxxxxx, computer ip xxx.xxx.xxx.xxx, which failed due to the incorrect output data path.
         - Ran \`ls -F static/\` in session xxxxxxxx, computer ip xxx.xxx.xxx.xxx and discovered image assets are stored as \`.webp\`.
         - Ran \`bash train.sh\` in session xxxxxxxx, computer ip xxx.xxx.xxx.xxx, it is still running now.
        -->
    </recent_actions>

    <experiment_history>
        <!-- A summary of the last few significant experiments and their outcomes. -->
        <!-- Example:
         - Experiment 1 (Hypothesis 1): Ran training with verbosity penalty. Result: Alignment score increased to 0.86, but helpfulness dropped slightly. See logs in `/workspace/task/logs/exp_1/`.
         - Experiment 2 (Hypothesis 2): Implemented reward normalization. Result: Training was stable, loss converged faster. Final score was 0.84. See logs in `/workspace/task/logs/exp_2/`.
        -->
    </experiment_history>

</state_snapshot>""",
        'required': True
    })

    def __str__(self) -> str:
        ret = f'**InternalSummarizeAction**\n'
        if self.summary_content:
            ret += f'Summary content: {self.summary_content}\n'
        return ret

@dataclass
class EvalAction(BaseAction):
    """An action where the agent evaluates the agent's output.

    Attributes:
        thought (str): The agent's explanation of its actions.
        action_type (str): The action type, namely ActionType.EVAL.
    """
    action_type: str = ActionType.EVAL
    description: str = """An action where the agent evaluates the agent's output (some files and the content inside the files), which is declared in the task description  (original task instead of subgoal). The argument of this action should be empty, do not add any key inside the argument"""

    @property
    def message(self) -> str:
        return f'The env is evaluating...: {self.thought}'
    
    def __str__(self) -> str:
        ret = f'**EvalAction**\n'
        if self.thought:
            ret += f'Thought: {self.thought}\n'
        return ret