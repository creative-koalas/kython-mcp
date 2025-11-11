"""
LLM Backend 基础抽象类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Iterator, List, Literal, Optional, Union


@dataclass
class Message:
    """聊天消息"""
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class LLMResponse:
    """LLM 响应结果"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None  # {"prompt_tokens": x, "completion_tokens": y, "total_tokens": z}
    finish_reason: Optional[str] = None  # "stop", "length", "content_filter", etc.
    raw_response: Optional[Dict] = None  # 原始 API 响应


class LLMBackend(ABC):
    """LLM 后端抽象基类"""

    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        同步完成对话

        Args:
            messages: 消息历史
            temperature: 采样温度 (0.0-2.0)
            max_tokens: 最大生成 token 数
            **kwargs: 其他模型特定参数

        Returns:
            LLMResponse 对象
        """
        pass

    @abstractmethod
    async def acomplete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        异步完成对话

        Args:
            messages: 消息历史
            temperature: 采样温度 (0.0-2.0)
            max_tokens: 最大生成 token 数
            **kwargs: 其他模型特定参数

        Returns:
            LLMResponse 对象
        """
        pass

    @abstractmethod
    def stream_complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        流式同步完成对话

        Args:
            messages: 消息历史
            temperature: 采样温度 (0.0-2.0)
            max_tokens: 最大生成 token 数
            **kwargs: 其他模型特定参数

        Yields:
            文本片段
        """
        pass

    @abstractmethod
    async def astream_complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        流式异步完成对话

        Args:
            messages: 消息历史
            temperature: 采样温度 (0.0-2.0)
            max_tokens: 最大生成 token 数
            **kwargs: 其他模型特定参数

        Yields:
            文本片段
        """
        pass
