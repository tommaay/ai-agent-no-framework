import openai
from abc import ABC, abstractmethod
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class AgentBase(ABC):
  """Base class for AI agents that defines the common interface."""

  def __init__(self, name, max_retries=2, verbose=True):
    self.name = name
    self.max_retries = max_retries
    self.verbose = verbose

  @abstractmethod
  def execute(self, *args, **kwargs):
    pass

  def call_openai(self, messages, tempurature=0.7, max_tokens=150):
    retries = 0
    while retries < self.max_retries:
      try:
        if self.verbose:
          logger.info(f"[{self.name}] sending message to OpenAi with messages: {messages}")
          for message in messages:
            logger.debug(f" {message['role']}: {msg['content']}")
        response = openai.chat.completions.create(
          model="gpt-4o",
          messages = messages,
          temperature=tempurature,
          max_tokens=max_tokens,
        )
        reply = response.choices[0].message
        if self.verbose:
          logger.info(f"[{self.name}] received response from OpenAi: {reply}")
        return reply
        except Exception as e:
          retries += 1
          logger.error(f"[{self.name}] error sending message to OpenAi: {e}. Retry attempt {retries}/{self.max_retries}")
    raise Exception(f"Failed to get response from OpenAi after {self.max_retries} retries")
