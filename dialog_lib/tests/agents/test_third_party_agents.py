import os
import pytest
from dialog_lib.agents.openai import DialogOpenAI
from dialog_lib.agents.anthropic import DialogAnthropic

def test_openai_agent():
    os.environ['OPENAI_API_KEY'] = 'sk_test_1234567890'
    agent = DialogOpenAI()
    assert agent is not None

def test_anthropic_agent():
    agent = DialogAnthropic()
    assert agent is not None