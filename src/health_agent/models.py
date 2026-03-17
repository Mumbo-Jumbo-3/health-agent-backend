from langchain_core.language_models import BaseChatModel

from health_agent.config import LLMProvider, Settings


def get_chat_model(settings: Settings) -> BaseChatModel:
    match settings.llm_provider:
        case LLMProvider.OPENAI:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
            )
        case LLMProvider.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=settings.anthropic_model,
                api_key=settings.anthropic_api_key,
            )
        case LLMProvider.XAI:
            from langchain_xai import ChatXAI

            return ChatXAI(
                model=settings.xai_model,
                api_key=settings.xai_api_key,
            )
