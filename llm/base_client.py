# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT


from abc import ABC, abstractmethod
from llm.config import ModelParameters
from llm.llm_basics import LLMMessage, LLMResponse
from research_gym.action import BaseAction

class BaseLLMClient(ABC):
    """Base class for LLM clients."""

    def __init__(self, model_parameters: ModelParameters):
        self.api_key: str = model_parameters.api_key
        self.base_url: str | None = model_parameters.base_url
        self.api_version: str | None = model_parameters.api_version
        self.model_parameters: ModelParameters = model_parameters

    @abstractmethod
    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        pass

    @abstractmethod
    def chat(self, messages: list[LLMMessage], tools: list[BaseAction] | None = None, reuse_history: bool = False) -> LLMResponse:
        """Send chat messages to the LLM."""
        pass

    @abstractmethod
    def supports_tool_calling(self) -> bool:
        """Check if the current model supports tool calling."""
        pass
