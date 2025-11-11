# LLM Backend

统一的大模型后端调用接口,支持 OpenAI 兼容 API。

## 特性

- ✅ OpenAI 官方 API 支持
- ✅ Azure OpenAI 支持
- ✅ 本地模型服务 (vLLM, Ollama, etc.)
- ✅ 同步/异步调用
- ✅ 流式输出
- ✅ 自动重试机制
- ✅ 使用 `.env` 管理配置

## 快速开始

### 1. 安装依赖

```bash
uv add openai python-dotenv
# 或
pip install openai python-dotenv
```

### 2. 配置环境变量

复制 `.env.example` 到 `.env`:

```bash
cp .env.example .env
```

编辑 `.env` 文件:

```bash
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo
```

### 3. 基本使用

```python
from llm_backend import Message, OpenAICompatibleBackend

# 初始化后端 (自动从 .env 加载配置)
backend = OpenAICompatibleBackend()

# 准备消息
messages = [
    Message(role="system", content="你是一个有帮助的助手。"),
    Message(role="user", content="你好!"),
]

# 同步调用
response = backend.complete(messages)
print(response.content)

# 流式调用
for chunk in backend.stream_complete(messages):
    print(chunk, end="", flush=True)
```

### 4. 异步使用

```python
import asyncio

async def main():
    backend = OpenAICompatibleBackend()

    messages = [
        Message(role="user", content="解释一下 Python asyncio"),
    ]

    # 异步调用
    response = await backend.acomplete(messages)
    print(response.content)

    # 异步流式调用
    async for chunk in backend.astream_complete(messages):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

## 支持的服务

### OpenAI 官方

```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4
```

### Azure OpenAI

```bash
OPENAI_API_KEY=your-azure-key
OPENAI_BASE_URL=https://your-resource.openai.azure.com/
OPENAI_MODEL=your-deployment-name
```

### vLLM (本地服务)

```bash
# 启动 vLLM 服务:
# vllm serve meta-llama/Llama-2-7b-chat-hf --port 8000

OPENAI_API_KEY=dummy-key
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_MODEL=meta-llama/Llama-2-7b-chat-hf
```

### Ollama (OpenAI 兼容端点)

```bash
# Ollama 默认端口 11434
OPENAI_API_KEY=dummy-key
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama2
```

## 高级用法

### 直接传入配置 (不使用 .env)

```python
backend = OpenAICompatibleBackend(
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
    model="gpt-4",
    load_env=False,  # 不加载 .env 文件
)
```

### 自定义参数

```python
response = backend.complete(
    messages=messages,
    temperature=0.8,
    max_tokens=500,
    top_p=0.95,
    frequency_penalty=0.5,
)
```

### 并发调用

```python
import asyncio

async def batch_query():
    backend = OpenAICompatibleBackend()

    questions = [
        "什么是 Python?",
        "什么是 asyncio?",
        "什么是 GIL?",
    ]

    tasks = [
        backend.acomplete([Message(role="user", content=q)])
        for q in questions
    ]

    results = await asyncio.gather(*tasks)
    return [r.content for r in results]

asyncio.run(batch_query())
```

## API 参考

### `OpenAICompatibleBackend`

#### 初始化参数

- `api_key`: API 密钥 (默认从 `OPENAI_API_KEY` 环境变量读取)
- `base_url`: API 基础 URL (默认从 `OPENAI_BASE_URL` 读取,或 `https://api.openai.com/v1`)
- `model`: 默认模型名称 (默认从 `OPENAI_MODEL` 读取,或 `gpt-3.5-turbo`)
- `timeout`: 请求超时时间,单位秒 (默认 60.0)
- `max_retries`: 最大重试次数 (默认 3)
- `load_env`: 是否自动加载 .env 文件 (默认 True)

#### 方法

- `complete(messages, temperature=0.7, max_tokens=None, **kwargs)`: 同步完成
- `acomplete(messages, temperature=0.7, max_tokens=None, **kwargs)`: 异步完成
- `stream_complete(messages, temperature=0.7, max_tokens=None, **kwargs)`: 同步流式完成
- `astream_complete(messages, temperature=0.7, max_tokens=None, **kwargs)`: 异步流式完成

### `LLMResponse`

返回结果对象:

```python
@dataclass
class LLMResponse:
    content: str  # 生成的文本
    model: str  # 使用的模型名称
    usage: Optional[Dict[str, int]]  # Token 使用统计
    finish_reason: Optional[str]  # 结束原因
    raw_response: Optional[Dict]  # 原始 API 响应
```

## 示例代码

查看 `example.py` 获取完整示例:

```bash
python src/llm_backend/example.py
```

## 扩展到其他后端

要添加新的后端 (如 Claude, Gemini),继承 `LLMBackend` 基类并实现抽象方法即可。参考 `openai_compatible.py` 的实现。
