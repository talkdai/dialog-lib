import pytest
from dialog_lib.agents.abstract import AbstractLLM


def test_abstract_agent_with_invalid_config():
    with pytest.raises(ValueError):
        AbstractLLM(config="invalid_config")


def test_abstract_agent_with_valid_config():
    config = {
        "model": "gpt3.5-turbo",
        "temperature": 0.5,
    }
    agent = AbstractLLM(config=config)
    assert agent.config == config
    assert agent.prompt is None
    assert agent.session_id is None
    assert agent.relevant_contents is None
    assert agent.dataset is None
    assert agent.llm_api_key is None
    assert agent.parent_session_id is None

def test_abstract_agent_get_prompt():
    config = {
        "model": "gpt3.5-turbo",
        "temperature": 0.5,
    }
    agent = AbstractLLM(config=config)

def test_abstract_agent_memory():
    config = {
        "model": "gpt3.5-turbo",
        "temperature": 0.5,
    }
    agent = AbstractLLM(config=config)
    with pytest.raises(NotImplementedError):
        agent.memory

def test_pre_and_post_processing():
    config = {
        "model": "gpt3.5-turbo",
        "temperature": 0.5,
    }
    agent = AbstractLLM(config=config)
    output = agent.preprocess(input="Hello")
    assert output == "Hello"

    post_processed_output = agent.postprocess(output="Hello")
    assert post_processed_output == "Hello"

def test_abstract_agent_generate_prompt():
    config = {
        "model": "gpt3.5-turbo",
        "temperature": 0.5,
    }
    agent = AbstractLLM(config=config)
    prompt = agent.generate_prompt(text="Hello")
    assert prompt == "Hello"

def test_abstract_agent_process(mocker):
    config = {
        "model": "gpt3.5-turbo",
        "temperature": 0.5,
        "prompt": {
            "fallback_not_found_relevant_contents": "404 Not Found"
        }
    }
    agent = AbstractLLM(config=config)

    agent.relevant_contents = [] # sample mock
    mocker.patch('dialog_lib.agents.abstract.AbstractLLM.llm')
    mocker.patch('dialog_lib.agents.abstract.AbstractLLM.llm.invoke', return_value={'text': '404 Not Found'})
    output = agent.process(input="Hello")
    assert output == {'text': '404 Not Found'}
