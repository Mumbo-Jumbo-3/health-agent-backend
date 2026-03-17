from health_agent.config import LLMProvider, Settings
from health_agent.models import get_chat_model


def test_openai_model():
    settings = Settings(llm_provider=LLMProvider.OPENAI, openai_api_key="test-key")
    model = get_chat_model(settings)
    from langchain_openai import ChatOpenAI

    assert isinstance(model, ChatOpenAI)


def test_anthropic_model():
    settings = Settings(llm_provider=LLMProvider.ANTHROPIC, anthropic_api_key="test-key")
    model = get_chat_model(settings)
    from langchain_anthropic import ChatAnthropic

    assert isinstance(model, ChatAnthropic)


def test_xai_model():
    settings = Settings(llm_provider=LLMProvider.XAI, xai_api_key="test-key")
    model = get_chat_model(settings)
    from langchain_xai import ChatXAI

    assert isinstance(model, ChatXAI)
