# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Azure client wrapper with tool integrations (OpenAI SDK >= 1.x)"""

import os
import time
import random
import json
from agents.utils.typing_compat import override

from openai import AzureOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionFunctionMessageParam,
)
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.shared_params.function_definition import FunctionDefinition

from llm.base_client import BaseLLMClient
from llm.llm_basics import LLMUsage, LLMMessage, LLMResponse
from llm.config import ModelParameters
from research_gym.action import BaseAction, ToolCall


class AzureClient(BaseLLMClient):
    """Azure client wrapper with tool schema generation."""

    def __init__(self, model_parameters: ModelParameters):
        super().__init__(model_parameters)

        # Read credentials and configuration
        if self.api_key == "":
            self.api_key: str = os.getenv("AZURE_API_KEY", "")
        if self.api_key == "":
            raise ValueError("Azure API key not provided. Set AZURE_API_KEY in environment variables or config file.")

        if self.base_url is None or self.base_url == "":
            self.base_url: str | None = os.getenv("AZURE_API_BASE_URL")
        if self.base_url is None:
            raise ValueError("Azure API base url not provided. Set AZURE_API_BASE_URL in environment variables or config file.")

        if self.api_version is None or self.api_version == "":
            self.api_version: str | None = os.getenv("AZURE_API_VERSION")
        if self.api_version is None:
            raise ValueError("Azure API version not provided.")

        # Azure client for the new SDK
        self.client: AzureOpenAI = AzureOpenAI(
            azure_endpoint=self.base_url,
            api_version=self.api_version,
            api_key=self.api_key,
        )
        
        self.message_history: list[ChatCompletionMessageParam] = []

    @override
    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        self.message_history = self.parse_messages(messages)

    @override
    def chat(
        self,
        messages: list[LLMMessage],
        tools: list[BaseAction] | None = None,
        reuse_history: bool = False
    ) -> LLMResponse:
        """Send chat messages to model provider with optional tool support."""
        azure_messages = self.parse_messages(messages)
        # Determine the actual messages to send based on reuse_history
        if reuse_history:
            self.message_history = self.message_history + azure_messages
            messages_to_send = self.message_history
        else:
            messages_to_send = azure_messages

        # Assemble tool schemas
        tool_schemas = None
        if tools:
            tool_schemas = [
                ChatCompletionToolParam(
                    type="function",
                    function=FunctionDefinition(
                        name=tool.action_type.value,
                        description=tool.description,
                        parameters=tool.get_input_schema(),
                    ),
                )
                for tool in tools
            ]

        # Self-adapting parameters (including differences for reasoning/o3 models)
        api_params = self._adapt_parameters_for_model(tool_schemas, messages_to_send)

        # Call and retry
        response = None
        error_message = ""

        for i in range(self.model_parameters.max_retries):
            try:
                # print(f"api_params.keys(): {api_params.keys()}")
                # print(f"model: {api_params['model']}")
                # print(f"n: {api_params['n']}")
                # print(f"tools: {api_params['tools']}")
                # print(f"reasoning_effort: {api_params['reasoning_effort']}")
                # print(f"tool_choice: {api_params['tool_choice']}")
                # print(f"temperature: {api_params['temperature']}")
                # print(f"top_p: {api_params['top_p']}")
                print("\n\nActual input: ", api_params["messages"], flush=True)
                response = self.client.chat.completions.create(**api_params)
                print("\n\nActual response: ", response)
                
                msg = getattr(response, "message", "")
                if "502" in msg:
                    continue
                break
            except Exception as e:
                error_message += f"Error {i + 1}: {str(e)}\n"
                time.sleep(random.randint(3, 30))
                continue

        if response is None:
            raise ValueError(f"Failed to get response from Azure after max retries: {error_message}")

        llm_response = self.parse_response(response)

        # Only append assistant's reply to internal history when reuse_history=True
        if reuse_history:
            if llm_response.tool_calls:
                self.message_history.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant",
                        content=llm_response.content,
                        tool_calls=[
                            ChatCompletionMessageToolCallParam(
                                id=tool_call.call_id,
                                type="function",
                                function=Function(
                                    name=tool_call.name,
                                    arguments=json.dumps(tool_call.arguments),
                                ),
                            )
                            for tool_call in llm_response.tool_calls
                        ],
                    )
                )
            elif llm_response.content:
                self.message_history.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant",
                        content=llm_response.content,
                    )
                )

        return llm_response

    @override
    def supports_tool_calling(self) -> bool:
        return True

    def _adapt_parameters_for_model(self, tool_schemas, messages_to_send) -> dict:
        """Adapt API parameters according to model type (compatible with reasoning/o3 series and regular chat models)"""
        model_name = (self.model_parameters.model or "").lower()

        # Basic parameters
        api_params: dict = {
            "model": self.model_parameters.model,
            "messages": messages_to_send,
            "n": 1,
        }
        if tool_schemas:
            api_params["tools"] = tool_schemas

        # reasoning_effort (if provided)
        # if getattr(self.model_parameters, "reasoning_effort", None):
        #     api_params["reasoning_effort"] = self.model_parameters.reasoning_effort
        if getattr(self.model_parameters, "tool_choice", "required"):
            api_params["tool_choice"] = self.model_parameters.tool_choice
        # For o3 and other reasoning models: use max_completion_tokens; avoid passing unsupported fields
        if "o3" in model_name:
            if self.model_parameters.max_tokens:
                api_params["max_completion_tokens"] = self.model_parameters.max_tokens
            if self.model_parameters.top_p is not None:
                api_params["top_p"] = self.model_parameters.top_p
            # Do not set parameters like temperature / parallel_tool_calls that are not supported in some reasoning models
        else:
            # Regular chat models
            if self.model_parameters.temperature is not None:
                api_params["temperature"] = self.model_parameters.temperature
            if self.model_parameters.top_p is not None:
                api_params["top_p"] = self.model_parameters.top_p
            # if self.model_parameters.max_tokens:
            #     api_params["max_tokens"] = self.model_parameters.max_tokens

        return api_params

    def parse_messages(self, messages: list[LLMMessage]) -> list[ChatCompletionMessageParam]:
        """Convert framework's LLMMessage to OpenAI ChatCompletionMessageParam"""
        azure_messages: list[ChatCompletionMessageParam] = []
        for msg in messages:
            if msg.tool_result:
                # Return tool execution results to the model
                azure_messages.append(
                    ChatCompletionToolMessageParam(
                        content=msg.tool_result.content,
                        role="tool",
                        tool_call_id=msg.tool_result.call_id,
                    )
                )
            elif msg.role == "system":
                if not msg.content:
                    raise ValueError("System message content is required")
                azure_messages.append(
                    ChatCompletionSystemMessageParam(
                        role="system",
                        content=msg.content,
                    )
                )
            elif msg.role == "user":
                if not msg.content:
                    raise ValueError("User message content is required")
                azure_messages.append(
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=msg.content,
                    )
                )
            elif msg.role == "assistant":
                # Assistant messages may have only tool_calls or only content, or both (here only content is retained; tool_calls are generated within this class)
                tool_call = None
                if msg.tool_call:
                    # tool_call = ChatCompletionFunctionMessageParam(
                    #     role="function",
                    #     name=msg.tool_call.name,
                    #     content=json.dumps({
                    #         "name": msg.tool_call.name,
                    #         "arguments": msg.tool_call.arguments
                    #     }),
                    # )
                    tool_call = ChatCompletionMessageToolCallParam(
                        id=msg.tool_call.call_id,
                        type="function",
                        function=Function(
                            name=msg.tool_call.name,
                            arguments=json.dumps(msg.tool_call.arguments),
                        ),
                    )
                azure_messages.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant",
                        content=msg.content,
                        tool_calls=[tool_call] if tool_call else None,
                    )
                )
            else:
                raise ValueError(f"Invalid message role: {msg.role}")
        return azure_messages

    def parse_response(self, response: ChatCompletion) -> LLMResponse:
        """Parse ChatCompletion (openai.types.chat.ChatCompletion) into framework's general return format"""

        choice = response.choices[0]

        # Tool call parsing
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

        # Usage parsing
        usage = None
        if response.usage:
            usage = LLMUsage(
                input_tokens=getattr(response.usage, "prompt_tokens", None),
                output_tokens=getattr(response.usage, "completion_tokens", None),
                total_tokens=getattr(response.usage, "total_tokens", None),
            )

        # Unified return
        llm_response = LLMResponse(
            role=choice.message.role,
            content=choice.message.content or "",
            reasoning=getattr(choice.message, "reasoning", None),
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            model=response.model,
            usage=usage,
        )
        return llm_response
