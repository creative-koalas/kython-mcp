"""
LLM Backend 模块

提供统一的大模型调用接口,支持 OpenAI 兼容 API
"""

from .base import LLMBackend, LLMResponse, Message
from .openai_compatible import OpenAICompatibleBackend

__all__ = [
    "LLMBackend",
    "LLMResponse",
    "Message",
    "OpenAICompatibleBackend",
]
