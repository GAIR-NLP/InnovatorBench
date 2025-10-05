"""
GLM (ZhipuAI) client based on official Python SDK (zai-sdk)
Aligns with standard OpenAI Chat Completions calling style and supports tool calling.

Reference documentation: https://docs.bigmodel.cn/cn/guide/develop/python/introduction
"""

import os
import json
import random
import time
from typing import List, Dict, Any

from agents.utils.typing_compat import override
from llm.base_client import BaseLLMClient
from llm.llm_basics import LLMUsage, LLMMessage, LLMResponse
from llm.config import ModelParameters
from research_gym.action import BaseAction, ToolCall


class GLMClient(BaseLLMClient):
    """ZhipuAI GLM client, mimicking StandardOpenAIClient interface and behavior"""

    def __init__(self, model_parameters: ModelParameters):
        super().__init__(model_parameters)

        self.api_key = self.api_key or os.getenv("ZAI_API_KEY", "") or os.getenv("ZHIPUAI_API_KEY", "")
        if self.api_key == "":
            raise ValueError("ZhipuAI API key not provided. Please set ZAI_API_KEY or ZHIPUAI_API_KEY.")

        try:
            from zai import ZhipuAiClient  # type: ignore
        except Exception as e:
            raise ImportError(
                "zai-sdk is not installed or unavailable. Please install first: pip install zai-sdk"
            ) from e

        # Initialize client. zai-sdk defaults to domestic open platform, pass base_url if custom endpoint needed
        if self.base_url:
            # Some SDK versions may not support base_url parameter, trying to pass according to latest documentation
            self.client = ZhipuAiClient(api_key=self.api_key, base_url=self.base_url)  # type: ignore
        else:
            self.client = ZhipuAiClient(api_key=self.api_key)  # type: ignore

        self.message_history: List[Dict[str, Any]] = []

    @override
    def set_chat_history(self, messages: List[LLMMessage]) -> None:
        self.message_history = self.parse_messages(messages)

    @override
    def chat(self, messages: list[LLMMessage], tools: list[BaseAction] | None = None, reuse_history: bool = False) -> LLMResponse:
        glm_messages = self.parse_messages(messages)

        if reuse_history:
            self.message_history = self.message_history + glm_messages
            messages_to_send = self.message_history
        else:
            messages_to_send = glm_messages

        tool_schemas = None
        if tools:
            tool_schemas = [
                {
                    "type": "function",
                    "function": {
                        # Keep consistent with StandardOpenAIClient, use action_type.value
                        "name": tool.action_type.value if hasattr(tool.action_type, "value") else tool.action_type,
                        "description": tool.description,
                        "parameters": tool.get_input_schema(),
                    },
                }
                for tool in tools
            ]

        api_params: Dict[str, Any] = {
            "model": self.model_parameters.model,
            "messages": messages_to_send,
        }

        if self.model_parameters.temperature is not None:
            api_params["temperature"] = self.model_parameters.temperature
        if self.model_parameters.top_p is not None:
            api_params["top_p"] = self.model_parameters.top_p
        if self.model_parameters.max_tokens:
            api_params["max_tokens"] = self.model_parameters.max_tokens
        if tool_schemas:
            api_params["tools"] = tool_schemas
            # ZhipuAI function calling defaults to auto, respects external input; defaults to auto if not set
            api_params["tool_choice"] = "auto" or self.model_parameters.tool_choice
            print(f"glm4.5's tool_choice: {api_params['tool_choice']}")
        if self.model_parameters.thinking:
            api_params["thinking"] = {"type": "enabled"}
            print(f"glm4.5's thinking: {api_params['thinking']}")
        print(tool_schemas)
        response = None
        error_message = ""

        # Call and retry
        for i in range(self.model_parameters.max_retries):
            try:
                # print("\n\nGLM actual input:", json.dumps({k: v for k, v in api_params.items() if k != 'messages'}, ensure_ascii=False))
                print("\n\n Actual messages: ", api_params["messages"], flush=True)
                response = self.client.chat.completions.create(**api_params, timeout=360.0)  # type: ignore
                print("\n\nGLM actual response:", response, flush=True)
                # if response.choices[0].message.tool_calls and response.choices[0].message.content != "":
                #     api_params["max_tokens"] = min(self.model_parameters.max_tokens * (i + 1),15000)
                #     continue
                break
            except Exception as e:
                error_message += f"Error {i + 1}: {str(e)}\n"
                time.sleep(random.randint(3, 15))
                continue

        if response is None:
            raise ValueError(f"Failed to get response from ZhipuAI after max retries: {error_message}")

        llm_response = self.parse_response(response)

        if reuse_history:
            if llm_response.tool_calls:
                self.message_history.append(
                    {
                        "role": "assistant",
                        "content": llm_response.content,
                        "tool_calls": [
                            {
                                "id": tool_call.call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.name,
                                    "arguments": json.dumps(tool_call.arguments),
                                },
                            }
                            for tool_call in llm_response.tool_calls
                        ],
                    }
                )
            elif llm_response.content:
                self.message_history.append({"role": "assistant", "content": llm_response.content})

        return llm_response

    @override
    def supports_tool_calling(self) -> bool:
        return True

    def parse_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        parsed_messages: List[Dict[str, Any]] = []
        for msg in messages:
            message: Dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.reasoning:
                message["reasoning"] = msg.reasoning
            if msg.tool_call:
                message["tool_calls"] = [
                    {
                        "id": msg.tool_call.call_id,
                        "type": "function",
                        "function": {
                            "name": msg.tool_call.name,
                            "arguments": json.dumps(msg.tool_call.arguments) if msg.tool_call.arguments else "{}",
                        },
                    }
                ]
            if msg.tool_result:
                message["name"] = msg.tool_result.name
                message["tool_call_id"] = msg.tool_result.call_id
                message["content"] = msg.tool_result.content

            parsed_messages.append(message)
        return parsed_messages

    def parse_response(self, response: Any) -> LLMResponse:  # response type defined by SDK
        # ZhipuAI response is compatible with OpenAI: choices[0].message
        choice = response.choices[0]

        tool_calls: List[ToolCall] | None = None
        message_obj = getattr(choice, "message", None) or getattr(choice, "delta", None)
        if message_obj is None:
            # Fallback: some implementations put message in choice dictionary
            message_obj = choice

        raw_tool_calls = getattr(message_obj, "tool_calls", None) or getattr(message_obj, "function_call", None)
        if raw_tool_calls:
            tool_calls = []
            # Unify into list structure
            iterable = raw_tool_calls if isinstance(raw_tool_calls, list) else [raw_tool_calls]
            for tc in iterable:
                func = getattr(tc, "function", None) or getattr(tc, "function_call", None) or getattr(tc, "functionCall", None)
                name = None
                arguments = {}
                call_id = getattr(tc, "id", "")
                if isinstance(tc, dict):
                    func = tc.get("function", tc.get("function_call", tc.get("functionCall")))
                    call_id = tc.get("id", call_id)
                if func:
                    if isinstance(func, dict):
                        name = func.get("name")
                        args_text = func.get("arguments", "{}")
                    else:
                        name = getattr(func, "name", None)
                        args_text = getattr(func, "arguments", "{}")
                    try:
                        arguments = json.loads(args_text) if isinstance(args_text, str) else (args_text or {})
                    except Exception:
                        arguments = {}
                if name:
                    tool_calls.append(
                        ToolCall(
                            name=name,
                            call_id=call_id or "tool_call_0",
                            arguments=arguments,
                        )
                    )

        usage = None
        usage_obj = getattr(response, "usage", None)
        if usage_obj:
            # Compatible with attribute or dictionary style
            if isinstance(usage_obj, dict):
                prompt_tokens = int(usage_obj.get("prompt_tokens", 0))
                completion_tokens = int(usage_obj.get("completion_tokens", 0))
                total_tokens = int(usage_obj.get("total_tokens", prompt_tokens + completion_tokens))
            else:
                prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0))
                completion_tokens = int(getattr(usage_obj, "completion_tokens", 0))
                total_tokens = int(getattr(usage_obj, "total_tokens", prompt_tokens + completion_tokens))
            usage = LLMUsage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        content = getattr(message_obj, "content", None)
        role = getattr(message_obj, "role", "assistant")
        finish_reason = getattr(choice, "finish_reason", None)
        model = getattr(response, "model", self.model_parameters.model)

        llm_response = LLMResponse(
            role=role,
            content=content or "",
            reasoning=getattr(message_obj, "reasoning", None),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            model=model,
            usage=usage,
        )
        return llm_response

    def _get_system_prompt(self, tools: list[BaseAction] | None = None) -> str:
        return """You are an expert AI software engineering agent.
Your primary goal is to resolve tasks by navigating the provided environment, understanding the problem, implementing robust solutions, and ensuring your changes are correct.

Follow these steps methodically:
1. Understand the Problem: Carefully read the task description
2. Explore and analyze the environment
3. Plan and execute your solution step by step
4. Verify and test your implementation
5. Complete the task when satisfied

Use the available tools effectively and think step by step."""


