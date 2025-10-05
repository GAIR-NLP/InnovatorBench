"""LLM Client wrapper for OpenAI and Anthropic APIs."""

from enum import Enum

from research_gym.action.action import BaseAction

from llm.config import ModelParameters
from llm.base_client import BaseLLMClient
from llm.llm_basics import LLMMessage, LLMResponse

class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    GLM = "glm"
    
def infer_provider_type(model_parameters: ModelParameters) -> LLMProvider:
        """Infer provider type based on model parameters"""
        base_url = model_parameters.base_url
        api_key = model_parameters.api_key
        model = model_parameters.model
        print(f"base_url: {base_url}")
        print(f"model: {model}")

        if base_url:
            base_url_lower = base_url.lower()
            # Infer provider type based on base_url
            if "gpt-5" in model or "azure" in base_url_lower or "azure-api" in base_url_lower:
                return LLMProvider.AZURE
            elif "anthropic" in base_url_lower or "api.myhispreadnlp.com" in base_url_lower:
                return LLMProvider.ANTHROPIC
            elif "openrouter.ai" in base_url_lower:
                # OpenRouter uses OpenAI compatible format
                return LLMProvider.OPENAI
            elif "openai" in base_url_lower:
                return LLMProvider.OPENAI
            elif "glm" in model:
                print("use glm")
                return LLMProvider.GLM
            elif "kimi" in model:
                print("use kimi")
                return LLMProvider.OPENAI
            else:
                # Other third-party APIs are usually compatible with OpenAI format
                return LLMProvider.OPENAI
        else:
            # If no base_url, infer based on API key and model name
            if api_key.startswith("sk-ant-"):
                return LLMProvider.ANTHROPIC
            elif api_key.startswith("sk-or-"):
                return LLMProvider.OPENAI  # OpenRouter is compatible with OpenAI
            else:
                # Infer based on model name
                model_name = model_parameters.model.lower()
                if "claude" in model_name:
                    return LLMProvider.ANTHROPIC
                elif "glm" in model_name or "zhipu" in model_name:
                    return LLMProvider.GLM
                elif "gpt" in model_name or "o1" in model_name or "o3" in model_name or "o4" in model_name:
                    return LLMProvider.AZURE  # Assume GPT models use Azure
                else:
                    return LLMProvider.OPENAI  # Default


class LLMClient:
    """Main LLM client that supports multiple providers."""

    def __init__(self, provider: str | LLMProvider, model_parameters: ModelParameters):
        if isinstance(provider, str):
            provider = LLMProvider(provider)

        self.provider: LLMProvider = provider

        if provider == LLMProvider.OPENAI:
            # Check if the model needs a standard OpenAI client
            if self._needs_standard_client(model_parameters):
                from .standard_openai_client import StandardOpenAIClient
                self.client = StandardOpenAIClient(model_parameters)
            else:
                from .openai_client import OpenAIClient
                self.client = OpenAIClient(model_parameters)
        elif provider == LLMProvider.ANTHROPIC:
            from .anthropic_client import AnthropicClient
            self.client = AnthropicClient(model_parameters)
        elif provider == LLMProvider.AZURE:
            from .azure_client import AzureClient
            self.client = AzureClient(model_parameters)
        elif provider == LLMProvider.GLM:
            from .glm_client import GLMClient
            self.client = GLMClient(model_parameters)

    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        self.client.set_chat_history(messages)

    def chat(self, messages: list[LLMMessage], tools: list[BaseAction] | None = None, reuse_history: bool = False) -> LLMResponse:
        """Send chat messages to the LLM."""
        return self.client.chat(messages, tools, reuse_history)

    def supports_tool_calling(self) -> bool:
        """Check if the current client supports tool calling."""
        return hasattr(self.client, 'supports_tool_calling') and self.client.supports_tool_calling()
    
    def _needs_standard_client(self, model_parameters: ModelParameters) -> bool:
        """Check if standard OpenAI client is needed"""
        # New Claude API requires standard client
        if "cr-claude" in model_parameters.model:
            return True
        
        if "kimi" in model_parameters.model:
            return True
        
        # If base_url contains specific third-party APIs, use standard client
        if model_parameters.base_url:
            third_party_urls = [
                "d3bcmskxf7dkfo.cloudfront.net",
                "openrouter.ai",
                "35.220.164.252",
                "ark.cn-beijing.volces.com",
                "http://Bedroc-Proxy-aUa5XAckaLer-1274981936.us-west-2.elb.amazonaws.com/api/v1",
            ]
            for url in third_party_urls:
                if url in model_parameters.base_url:
                    return True
        
        return False
