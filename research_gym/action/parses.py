from dataclasses import dataclass, field
from email.policy import default
from typing import ClassVar, Optional

from research_gym.action.action import BaseAction, ActionSecurityRisk
from research_gym.schema.action import ActionType


@dataclass
class ParsePdfAction(BaseAction):
    """Parse PDF file and extract text content.
    
    Attributes:
        file_path (str): PDF file path.
        save_path (str): File path to save the parsing result.
        thought (str): Reason or purpose for parsing.
    """

    action_type: ActionType = ActionType.PARSE_PDF
    file_path: str = field(
        default="",
        metadata={
            'tool_param': True,
            'description': 'The path to the PDF file to parse.',
            'required': True
        }
    )
    save_path: str = field(
        default="",
        metadata={
            'tool_param': True,
            'description': 'The path to save the parsed content.',
            'required': True
        }
    )
    thought: str = ''
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None
    description: str = "Parse a PDF file, extract text content and save to a file."

    @property
    def message(self) -> str:
        return f'Parsing PDF file: {self.file_path} to {self.save_path}'

    def __str__(self) -> str:
        ret = f'**ParsePdfAction**\n'
        if self.file_path:
            ret += f'File: {self.file_path}\n'
        if self.save_path:
            ret += f'Save to: {self.save_path}\n'
        if self.thought:
            ret += f'Thought: {self.thought}\n'
        return ret


@dataclass
class ParseDocxAction(BaseAction):
    """Parse DOCX file and extract text content.
    
    Attributes:
        file_path (str): DOCX file path.
        save_path (str): File path to save the parsing result.
        thought (str): Reason or purpose for parsing.
    """

    action_type: ActionType = ActionType.PARSE_DOCX
    file_path: str = field(
        default="",
        metadata={
            'tool_param': True,
            'description': 'The path to the DOCX file to parse.',
            'required': True
        }
    )
    save_path: str = field(
        default="",
        metadata={
            'tool_param': True,
            'description': 'The path to save the parsed content.',
            'required': True
        }
    )
    thought: str = ''
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None
    description: str = "Parse a DOCX file and save the parsed content to a file."

    @property
    def message(self) -> str:
        return f'Parsing DOCX file: {self.file_path} to {self.save_path}'

    def __str__(self) -> str:
        ret = f'**ParseDocxAction**\n'
        if self.file_path:
            ret += f'File: {self.file_path}\n'
        if self.save_path:
            ret += f'Save to: {self.save_path}\n'
        if self.thought:
            ret += f'Thought: {self.thought}\n'
        return ret


@dataclass
class ParseLatexAction(BaseAction):
    """Parse LaTeX file and extract text content.
    
    Attributes:
        file_path (str): LaTeX file path.
        save_path (str): File path to save the parsing result.
        thought (str): Reason or purpose for parsing.
    """

    action_type: ActionType = ActionType.PARSE_LATEX
    file_path: str = field(
        default="",
        metadata={
            'tool_param': True,
            'description': 'The path to the LaTeX file to parse.',
            'required': True
        }
    )
    save_path: str = field(
        default="",
        metadata={
            'tool_param': True,
            'description': 'The path to save the parsed content.',
            'required': True
        }
    )
    thought: str = ''
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None
    description: str = "Parse a LaTeX file and save the parsed content to a file."

    @property
    def message(self) -> str:
        return f'Parsing LaTeX file: {self.file_path} to {self.save_path}'

    def __str__(self) -> str:
        ret = f'**ParseLatexAction**\n'
        if self.file_path:
            ret += f'File: {self.file_path}\n'
        if self.save_path:
            ret += f'Save to: {self.save_path}\n'
        if self.thought:
            ret += f'Thought: {self.thought}\n'
        return ret


@dataclass
class ParseAudioAction(BaseAction):
    """Parse audio file and transcribe its content.
    
    Attributes:
        file_path (str): Audio file path.
        save_path (str): File path to save the parsing result.
        model (str): The model to use for audio transcription, default is 'whisper-1'.
        thought (str): Reason or purpose for parsing.
    """

    action_type: ActionType = ActionType.PARSE_AUDIO
    file_path: str = field(
        default="",
        metadata={
            'tool_param': True,
            'description': 'The path to the audio file to parse.',
            'required': True
        }
    )
    save_path: str = field(
        default="",
        metadata={
            'tool_param': True,
            'description': 'The path to save the parsed content.',
            'required': True
        }
    )
    model: str = field(
        default='whisper-1',
        metadata={
            'tool_param': True,
            'description': 'The model to use for audio transcription.',
            'required': False
        }
    )
    thought: str = ''
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None
    description: str = "Parse an audio file, transcribe its content and save the parsed content to a file."

    @property
    def message(self) -> str:
        return f'Parsing audio file: {self.file_path} with model {self.model} to {self.save_path}'

    def __str__(self) -> str:
        ret = f'**ParseAudioAction**\n'
        if self.file_path:
            ret += f'File: {self.file_path}\n'
        if self.save_path:
            ret += f'Save to: {self.save_path}\n'
        if self.model:
            ret += f'Model: {self.model}\n'
        if self.thought:
            ret += f'Thought: {self.thought}\n'
        return ret


@dataclass
class ParseImageAction(BaseAction):
    """Parse image file and analyze its content.
    
    Attributes:
        file_path (str): Image file path.
        save_path (str): File path to save the parsing result.
        task (str): Image analysis task description, default is to describe the image as detail as possible.
        thought (str): Reason or purpose for parsing.
    """

    action_type: ActionType = ActionType.PARSE_IMAGE
    file_path: str = field(
        default="",
        metadata={
            'tool_param': True,
            'description': 'The path to the image file to parse.',
            'required': True
        }
    )
    save_path: str = field(
        default="",
        metadata={
            'tool_param': True,
            'description': 'The path to save the parsed content.',
            'required': True
        }
    )
    task: str = field(
        default='Describe this image as detail as possible.',
        metadata={
            'tool_param': True,
            'description': 'The task description for image analysis.',
            'required': False
        }
    )
    thought: str = ''
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None
    description: str = "Parse an image file, analyze its content and save the parsed content to a file."

    @property
    def message(self) -> str:
        return f'Parsing image file: {self.file_path} to {self.save_path}'

    def __str__(self) -> str:
        ret = f'**ParseImageAction**\n'
        if self.file_path:
            ret += f'File: {self.file_path}\n'
        if self.save_path:
            ret += f'Save to: {self.save_path}\n'
        if self.task:
            ret += f'Task: {self.task}\n'
        if self.thought:
            ret += f'Thought: {self.thought}\n'
        return ret


@dataclass
class ParseVideoAction(BaseAction):
    """Parse video file and analyze its content.
    
    Attributes:
        file_path (str): Video file path.
        save_path (str): File path to save the parsing result.
        task (str): Video analysis task description, default is to describe the video as detail as possible.
        frame_interval (int): Frame interval, default is 30.
        thought (str): Reason or purpose for parsing.
    """

    action_type: ActionType = ActionType.PARSE_VIDEO
    file_path: str = field(
        default="",
        metadata={
            'tool_param': True,
            'description': 'The path to the video file to parse.',
            'required': True
        }
    )
    save_path: str = field(
        default="",
        metadata={
            'tool_param': True,
            'description': 'The path to save the parsed content.',
            'required': True
        }
    )
    task: str = field(
        default='Describe this image as detail as possible.',
        metadata={
            'tool_param': True,
            'description': 'The task description for video analysis.',
            'required': False
        }
    )
    frame_interval: int = field(
        default=30,
        metadata={
            'tool_param': True,
            'description': 'The frame interval for video analysis.',
            'required': False
        }
    )
    thought: str = ''
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None
    description: str = "Parse a video file, analyze its content and save the parsed content to a file."

    @property
    def message(self) -> str:
        return f'Parsing video file: {self.file_path} with frame interval {self.frame_interval} to {self.save_path}'

    def __str__(self) -> str:
        ret = f'**ParseVideoAction**\n'
        if self.file_path:
            ret += f'File: {self.file_path}\n'
        if self.save_path:
            ret += f'Save to: {self.save_path}\n'
        if self.task:
            ret += f'Task: {self.task}\n'
        if self.frame_interval:
            ret += f'Frame Interval: {self.frame_interval}\n'
        if self.thought:
            ret += f'Thought: {self.thought}\n'
        return ret


@dataclass
class ParsePptxAction(BaseAction):
    """Parse PPTX file and extract text content.
    
    Attributes:
        file_path (str): PPTX file path.
        save_path (str): File path to save the parsing result.
        thought (str): Reason or purpose for parsing.
    """

    action_type: ActionType = ActionType.PARSE_PPTX
    file_path: str = field(
        default="",
        metadata={
            'tool_param': True,
            'description': 'The path to the PPTX file to parse.',
            'required': True
        }
    )
    save_path: str = field(
        default="",
        metadata={
            'tool_param': True,
            'description': 'The path to save the parsed content.',
            'required': True
        }
    )
    thought: str = ''
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None
    description: str = "Parse a PPTX file and extract text content"

    @property
    def message(self) -> str:
        return f'Parsing PPTX file: {self.file_path} to {self.save_path}'

    def __str__(self) -> str:
        ret = f'**ParsePptxAction**\n'
        if self.file_path:
            ret += f'File: {self.file_path}\n'
        if self.save_path:
            ret += f'Save to: {self.save_path}\n'
        if self.thought:
            ret += f'Thought: {self.thought}\n'
        return ret
