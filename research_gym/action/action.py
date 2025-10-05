from dataclasses import dataclass, fields, MISSING, field
from enum import Enum
from typing import ClassVar, Dict, Any, List, Optional, Union, get_type_hints
import asyncio
from typing_extensions import override
from research_gym.base_node import BaseNode

ToolCallArguments = dict[str, str | int | float | dict[str, object] | list[object] | None]

@dataclass
class ToolCall:
    """Represents a parsed tool call."""
    call_id: str
    name: str
    arguments: ToolCallArguments = field(default_factory=dict)

    @override
    def __str__(self) -> str:
        return f"ToolCall(name={self.name}, arguments={self.arguments}, call_id={self.call_id})"

@dataclass
class ToolResult:
    """Result of a tool execution."""
    call_id: str
    name: str
    content: str | None = None

class ActionConfirmationStatus(str, Enum):
    CONFIRMED = 'confirmed'
    REJECTED = 'rejected'
    AWAITING_CONFIRMATION = 'awaiting_confirmation'


class ActionSecurityRisk(int, Enum):
    UNKNOWN = -1
    LOW = 0
    MEDIUM = 1
    HIGH = 2


@dataclass
class BaseAction(BaseNode):
    call_id: str = ''
    action_type: str = 'base'
    thought: str = ''
    runnable: ClassVar[bool] = False
    description: str = ''
    
    @classmethod
    def get_tool_parameters(cls) -> List[Dict[str, Any]]:
        """Get the tool parameters for LLM (auto-extracted from dataclass fields)."""
        parameters = []
        
        # Get all dataclass fields
        for field in fields(cls):
            # Check if field is marked as tool parameter in metadata
            if field.metadata.get('tool_param', False):
                param_dict = {
                    'name': field.name,
                    'type': cls._get_field_type_string(field),
                    'description': field.metadata.get('description', f'The {field.name} parameter'),
                    'required': field.metadata.get('required', cls._is_field_required(field))
                }
                
                # Add optional parameter attributes
                if 'enum' in field.metadata:
                    param_dict['enum'] = field.metadata['enum']
                if 'items' in field.metadata:
                    param_dict['items'] = field.metadata['items']
                
                parameters.append(param_dict)
        
        return parameters
    
    @classmethod
    def _is_field_required(cls, field) -> bool:
        """Check if a dataclass field is required (has no default value)."""
        return field.default == MISSING and field.default_factory == MISSING
    
    @classmethod
    def _get_field_type_string(cls, field) -> str:
        """Convert field type to JSON schema type string."""
        import typing
        
        # Get field type annotations
        field_type = field.type
        
        # Handle Optional types
        if hasattr(typing, 'get_origin') and typing.get_origin(field_type) is Union:
            args = typing.get_args(field_type)
            if len(args) == 2 and type(None) in args:
                # This is Optional[T], get the type of T
                field_type = args[0] if args[1] is type(None) else args[1]
        
        # Map Python types to JSON Schema types
        if field_type == str:
            return "string"
        elif field_type == int:
            return "integer"
        elif field_type == float:
            return "number"
        elif field_type == bool:
            return "boolean"
        elif hasattr(typing, 'get_origin') and typing.get_origin(field_type) is list:
            return "array"
        elif hasattr(typing, 'get_origin') and typing.get_origin(field_type) is dict:
            return "object"
        else:
            # Default return string
            return "string"
    
    @classmethod
    def json_definition(cls) -> Dict[str, Any]:
        """Generate OpenAI function calling definition."""
        return {
            "name": cls.action_type,
            "description": cls.description,
            "parameters": cls.get_input_schema()
        }
    
    @classmethod
    def get_input_schema(cls) -> Dict[str, Any]:
        """Get the input schema for the tool."""
        schema: Dict[str, Any] = {
            "type": "object",
        }

        properties: Dict[str, Dict[str, Any]] = {}
        required: List[str] = []

        for param in cls.get_tool_parameters():
            properties[param['name']] = {
                "type": param['type'],
                "description": param['description']
            }
            if param.get('enum'):
                properties[param['name']]["enum"] = param['enum']

            if param.get('items'):
                properties[param['name']]["items"] = param['items']

            if param.get('required', False):
                required.append(param['name'])

        # if properties == {}:
        #     return {}
        schema["properties"] = properties
        if len(required) > 0:
            schema["required"] = required

        return schema
    
    @classmethod
    def from_tool_arguments(cls, arguments: ToolCallArguments) -> 'BaseAction':
        """Create action instance from tool arguments."""
        # Basic implementation - subclasses should override this method
        return cls()

    # def to_dict(self) -> Dict[str, Any]:
    #     """Convert action to dictionary."""
    #     return {f.name: getattr(self, f.name) for f in fields(self)}