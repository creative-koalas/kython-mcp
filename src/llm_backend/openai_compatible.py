"""
OpenAI 兼容 API 后端实现

支持所有遵循 OpenAI API 规范的服务,如:
- OpenAI 官方 API
- Azure OpenAI
- vLLM
- Ollama (使用 OpenAI 兼容端点)
- 其他本地/云端 LLM 服务
"""

import os
from typing import AsyncIterator, Iterator, List, Optional

from dotenv import load_dotenv

from .base import LLMBackend, LLMResponse, Message


class OpenAICompatibleBackend(LLMBackend):
    """OpenAI 兼容 API 后端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        load_env: bool = True,
    ):
        """
        初始化 OpenAI 兼容后端

        Args:
            api_key: API 密钥 (默认从环境变量 OPENAI_API_KEY 读取)
            base_url: API 基础 URL (默认从环境变量 OPENAI_BASE_URL 读取)
            model: 默认模型名称 (默认从环境变量 OPENAI_MODEL 读取)
            timeout: 请求超时时间(秒)
            max_retries: 最大重试次数
            load_env: 是否自动加载 .env 文件
        """
        if load_env:
            load_dotenv()

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError(
                "API key 未设置。请通过参数传入或设置环境变量 OPENAI_API_KEY"
            )

        # 延迟导入 OpenAI 客户端
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError(
                "请安装 openai 库: pip install openai 或 uv add openai"
            )

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

        self._async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def _messages_to_openai_format(self, messages: List[Message]) -> List[dict]:
        """将消息列表转换为 OpenAI API 格式"""
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        同步完成对话

        Args:
            messages: 消息历史
            temperature: 采样温度 (0.0-2.0)
            max_tokens: 最大生成 token 数
            model: 模型名称 (默认使用初始化时的模型)
            **kwargs: 其他 OpenAI API 参数 (top_p, frequency_penalty, etc.)

        Returns:
            LLMResponse 对象
        """
        response = self._client.chat.completions.create(
            model=model or self.model,
            messages=self._messages_to_openai_format(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            **kwargs
        )

        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if response.usage else None,
            finish_reason=choice.finish_reason,
            raw_response=response.model_dump(),
        )

    async def acomplete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        异步完成对话

        Args:
            messages: 消息历史
            temperature: 采样温度 (0.0-2.0)
            max_tokens: 最大生成 token 数
            model: 模型名称 (默认使用初始化时的模型)
            **kwargs: 其他 OpenAI API 参数

        Returns:
            LLMResponse 对象
        """
        response = await self._async_client.chat.completions.create(
            model=model or self.model,
            messages=self._messages_to_openai_format(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            **kwargs
        )

        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if response.usage else None,
            finish_reason=choice.finish_reason,
            raw_response=response.model_dump(),
        )

    def stream_complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        流式同步完成对话

        Args:
            messages: 消息历史
            temperature: 采样温度 (0.0-2.0)
            max_tokens: 最大生成 token 数
            model: 模型名称 (默认使用初始化时的模型)
            **kwargs: 其他 OpenAI API 参数

        Yields:
            文本片段
        """
        stream = self._client.chat.completions.create(
            model=model or self.model,
            messages=self._messages_to_openai_format(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def astream_complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        流式异步完成对话

        Args:
            messages: 消息历史
            temperature: 采样温度 (0.0-2.0)
            max_tokens: 最大生成 token 数
            model: 模型名称 (默认使用初始化时的模型)
            **kwargs: 其他 OpenAI API 参数

        Yields:
            文本片段
        """
        stream = await self._async_client.chat.completions.create(
            model=model or self.model,
            messages=self._messages_to_openai_format(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
