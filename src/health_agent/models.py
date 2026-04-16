from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_xai import ChatXAI

from health_agent.config import Settings


def get_trusted_grok_model(settings: Settings) -> ChatXAI:
    return ChatXAI(
        model=settings.trusted_xai_model,
        api_key=settings.xai_api_key,
    )


def get_unrestricted_grok_model(settings: Settings) -> ChatXAI:
    return ChatXAI(
        model=settings.unrestricted_xai_model,
        api_key=settings.xai_api_key,
    )


def get_claude_synthesis_model(settings: Settings) -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.anthropic_synthesis_model,
        api_key=settings.anthropic_api_key,
    )


def get_claude_judge_model(settings: Settings) -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.anthropic_judge_model,
        api_key=settings.anthropic_api_key,
    )


def get_embeddings_model(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
