"""
Standard OpenAI compatible client
Supports standard chat.completions.create API format
"""

import os
import json
import random
import time
import openai
from typing import List, Dict, Any

from agents.utils.typing_compat import override
from llm.base_client import BaseLLMClient
from llm.llm_basics import LLMUsage, LLMMessage, LLMResponse
from llm.config import ModelParameters
from research_gym.action import BaseAction, ToolCall


class StandardOpenAIClient(BaseLLMClient):
    """Standard OpenAI compatible client, supports third-party Claude API"""

    def __init__(self, model_parameters: ModelParameters):
        """
        Initialize the StandardOpenAIClient.
        """
        super().__init__(model_parameters)

        if self.api_key == "":
            self.api_key: str = os.getenv("OPENAI_API_KEY", "")

        if self.api_key == "":
            raise ValueError("API key not provided. Set in environment variables or config file.")

        # Use custom base_url if provided
        if model_parameters.base_url:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=model_parameters.base_url
            )
        else:
            self.client = openai.OpenAI(api_key=self.api_key)
        
        self.message_history: List[Dict[str, Any]] = []
        self.add_system_prompt = True

    @override
    def set_chat_history(self, messages: List[LLMMessage]) -> None:
        """Set chat history"""
        self.message_history = self.parse_messages(messages)

    @override
    def chat(self, messages: list[LLMMessage], tools: list[BaseAction] | None = None, reuse_history: bool = False) -> LLMResponse:
        """Send chat messages to model provider"""
        openai_messages = self.parse_messages(messages)
        
        # Core fix: when reuse_history=False, use provided messages instead of appending to history
        if reuse_history:
            # Append to existing history
            self.message_history = self.message_history + openai_messages
            messages_to_send = self.message_history
        else:
            # Use provided messages, do not modify internal history
            messages_to_send = openai_messages

        # Prepare tool schemas
        tool_schemas = None
        if tools:
            # for tool in tools:
            #     print(f"tool: {tool.action_type}, type: {type(tool.action_type)}")
            tool_schemas = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.action_type.value,
                        "description": tool.description,
                        "parameters": tool.get_input_schema()
                    }
                }
                for tool in tools
            ]
            print("tool_schemas: ", tool_schemas)

        # Prepare API parameters
        api_params = {
            "model": self.model_parameters.model,
            "messages": messages_to_send,
        }
        
        # Add optional parameters
        if self.model_parameters.temperature is not None:
            api_params["temperature"] = self.model_parameters.temperature
        if self.model_parameters.top_p is not None:
            api_params["top_p"] = self.model_parameters.top_p
        if self.model_parameters.max_tokens:
            api_params["max_tokens"] = self.model_parameters.max_tokens
        if tool_schemas:
            api_params["tools"] = tool_schemas
            api_params["tool_choice"] = "required"
        # if self.model_parameters.parallel_tool_calls is not None:
        #     api_params["parallel_tool_calls"] = self.model_parameters.parallel_tool_calls

        # For specific Claude APIs, add thinking mode support
        if "cr-claude" in self.model_parameters.model or "claude" in self.model_parameters.model:
            # api_params["reasoning"] = {"max_tokens": 2000}
            pass
        # api_params["extra_body"] = {
        #     "thinking": {"type": "enabled", "budget_tokens": 2000}
        # }

        response = None
        error_message = ""
        
        for i in range(self.model_parameters.max_retries):
            try:
                # print("\n\nActual input: ", json.dumps(api_params, indent=4, ensure_ascii=False))
                # print("\n\nActual input: ", api_params, flush=True)
                print("\n\nActual input: ", api_params["messages"], flush=True)
                response = self.client.chat.completions.create(**api_params,timeout=360.0)
                print("\n\nActual response: ", response, flush=True)
                # if response.choices[0].message.content:
                #     print("\n\nActual response: ", response.choices[0].message.content, flush=True)
                if not response.choices[0].message.tool_calls and response.choices[0].message.content != "":
                    api_params["max_tokens"] = min(self.model_parameters.max_tokens * (i + 1),15000)
                    continue
                #     print("\n\nTool call id: ", response.choices[0].message.tool_calls[0].id, flush=True)
                #     print("\n\nTool call name: ", response.choices[0].message.tool_calls[0].function.name, flush=True)
                # raise ValueError("Stop here")
                break
            except Exception as e:
                error_message += f"Error {i + 1}: {str(e)}\n"
                # Random wait 3-30 seconds
                time.sleep(random.randint(3, 30))
                continue

        if response is None:
            raise ValueError(f"Failed to get response from OpenAI after max retries: {error_message}")

        llm_response = self.parse_response(response)

        # Core fix: only update internal history when reuse_history=True
        if reuse_history:
            if llm_response.tool_calls:
                self.message_history.append({
                    "role": "assistant",
                    "content": llm_response.content,
                    "tool_calls": [
                        {
                            "id": tool_call.call_id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": json.dumps(tool_call.arguments)
                            }
                        }
                        for tool_call in llm_response.tool_calls
                    ]
                })
            elif llm_response.content:
                self.message_history.append({
                    "role": "assistant",
                    "content": llm_response.content
                })

        return llm_response

    @override
    def supports_tool_calling(self) -> bool:
        """Check if the current model supports tool calling"""
        # Most modern models support tool calling
        return True

    def parse_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Parse messages into OpenAI format"""
        parsed_messages = []
        
        for msg in messages:
            message = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.reasoning:
                message["reasoning"] = msg.reasoning
            if msg.tool_call:
                message["tool_calls"] = [{
                    "id": msg.tool_call.call_id,
                    "type": "function",
                    "function": {
                        "name": msg.tool_call.name,
                        "arguments": json.dumps(msg.tool_call.arguments)
                    }
                }]
            if msg.tool_result:
                message["name"] = msg.tool_result.name
                message["tool_call_id"] = msg.tool_result.call_id
                message["content"] = msg.tool_result.content

            parsed_messages.append(message)
        
        return parsed_messages 


    def parse_response(self, response: openai.ChatCompletion) -> LLMResponse:
        """Parse response"""
        choice = response.choices[0]

        # Parse tool calls
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = []
            for tool_call in choice.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tool_call.function.name,
                        call_id=tool_call.id,
                        arguments=json.loads(tool_call.function.arguments) if tool_call.function.arguments else {},
                    )
                )
        
        # Parse usage
        usage = None
        if response.usage:
            usage = LLMUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        # Create LLM response
        llm_response = LLMResponse(
            role=choice.message.role,
            content=choice.message.content or "",
            reasoning=getattr(choice.message, "reasoning", None),  # Non-standard field, safe access
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            model=response.model,
            usage=usage,
        )

        return llm_response
        

    def _get_system_prompt(self, tools: list[BaseAction] | None = None) -> str:
        """Get the system prompt."""
        # Use fixed root node prompt as default system prompt to avoid complex dependencies
        return """You are an expert AI software engineering agent.
Your primary goal is to resolve tasks by navigating the provided environment, understanding the problem, implementing robust solutions, and ensuring your changes are correct.

Follow these steps methodically:
1. Understand the Problem: Carefully read the task description
2. Explore and analyze the environment
3. Plan and execute your solution step by step
4. Verify and test your implementation
5. Complete the task when satisfied

Use the available tools effectively and think step by step.""" 
