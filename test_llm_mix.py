"""
LLM_MIX MCP 服务测试套件

运行方式:
    pytest test_llm_mix.py -v

依赖:
    pip install pytest pytest-asyncio fastmcp openai

这不是普通脚本,不要用 `python test_llm_mix.py` 直接运行。
pytest 会自动发现下面以 test_ 开头的函数,并逐个执行。

所有测试都不联网: OpenAI / DeepSeek / Kimi / Qwen 调用都会被 mock 掉。
这样测试可以离线运行,也不会消耗 API 额度。
"""
import httpx
import pytest
from unittest.mock import patch, MagicMock, Mock

# 被测对象。文件名是 LLM_MIX.py, 所以这里直接 import LLM_MIX。
import LLM_MIX
from fastmcp import Client
from openai import APIConnectionError


@pytest.fixture(autouse=True)
def _clean_state():

    LLM_MIX.SESSIONS.clear()
    LLM_MIX.SESSION_LOCKS.clear()
    yield
    LLM_MIX.SESSIONS.clear()
    LLM_MIX.SESSION_LOCKS.clear()


def _conn_error():

    return APIConnectionError(request=httpx.Request("POST", "https://example.com"))


def _fake_openai(text="reply"):

    client = MagicMock()
    client.chat.completions.create.return_value.choices[0].message.content = text
    return client


async def _call_tool(name, args):

    async with Client(LLM_MIX.mcp) as client:
        result = await client.call_tool(name, args)
        return result.data


class TestTruncate:

    def test_short_unchanged(self):
        assert LLM_MIX._truncate("hello") == "hello"

    def test_none_and_empty(self):
        assert LLM_MIX._truncate(None) is None
        assert LLM_MIX._truncate("") == ""

    def test_exact_limit_unchanged(self):
        s = "a" * 100
        assert LLM_MIX._truncate(s, max_chars=100) == s

    def test_over_limit_truncated(self):
        s = "a" * 50 + "b" * 50 + "c" * 50  # 150 字符
        out = LLM_MIX._truncate(s, max_chars=100)
        assert len(out) < len(s)
        assert "省略" in out
        assert out.startswith("a") and out.endswith("c")  # 保留头尾


class TestCheckHistory:

    def test_under_limit_returns_all(self):
        h = [{"role": "user", "content": str(i)} for i in range(5)]
        assert LLM_MIX._check_history(h) == h

    def test_over_limit_no_system(self):
        h = [{"role": "user", "content": str(i)} for i in range(30)]
        out = LLM_MIX._check_history(h)
        assert len(out) == LLM_MIX.MAX_HISTORY_MESSAGES
        assert out[-1]["content"] == "29"  # 保留最新的

    def test_over_limit_keeps_system_first(self):
        h = [{"role": "system", "content": "sys"}]
        h += [{"role": "user", "content": str(i)} for i in range(30)]
        out = LLM_MIX._check_history(h)
        assert len(out) == LLM_MIX.MAX_HISTORY_MESSAGES
        assert out[0]["role"] == "system"   # system 必须留在第一位
        assert out[-1]["content"] == "29"


class TestExtractCodexSessionId:

    def test_valid_uuid(self):
        sid = "12345678-1234-1234-1234-123456789abc"
        assert LLM_MIX._extract_codex_session_id(f"xx session id: {sid} yy") == sid

    def test_missing(self):
        assert LLM_MIX._extract_codex_session_id("没有 id") is None

    def test_malformed(self):
        assert LLM_MIX._extract_codex_session_id("session id: not-a-uuid") is None


class TestSessionLock:

    def test_same_sid_same_lock(self):
        assert LLM_MIX._get_session_lock("x") is LLM_MIX._get_session_lock("x")

    def test_diff_sid_diff_lock(self):
        assert LLM_MIX._get_session_lock("a") is not LLM_MIX._get_session_lock("b")


class TestCallWithRetry:

    def test_success_first_try(self):
        f = Mock(return_value="ok")
        assert LLM_MIX._call_with_retry(f) == "ok"
        assert f.call_count == 1

    def test_retry_then_success(self):
        f = Mock(side_effect=[_conn_error(), _conn_error(), "ok"])
        # patch sleep 避免测试真的等待 1s/2s。
        with patch("LLM_MIX.time.sleep") as slp:
            assert LLM_MIX._call_with_retry(f, max_attempts=3) == "ok"
        assert f.call_count == 3
        assert slp.call_count == 2

    def test_exhausted_raises(self):
        f = Mock(side_effect=[_conn_error()] * 3)
        with patch("LLM_MIX.time.sleep"):
            with pytest.raises(APIConnectionError):
                LLM_MIX._call_with_retry(f, max_attempts=3)

    def test_non_retryable_propagates_immediately(self):
        f = Mock(side_effect=ValueError("boom"))
        with pytest.raises(ValueError):
            LLM_MIX._call_with_retry(f, max_attempts=3)
        assert f.call_count == 1  # 不可重试的错误立即抛出，不重试



class TestCall:
    def test_basic_reply(self):
        client = _fake_openai("hi")
        with patch.object(LLM_MIX, "OpenAI", return_value=client):
            assert LLM_MIX._call("deepseek", "q", None, 0.5) == "hi"
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.5
        assert kwargs["messages"][-1] == {"role": "user", "content": "q"}

    def test_system_prompt_prepended(self):
        client = _fake_openai()
        with patch.object(LLM_MIX, "OpenAI", return_value=client):
            LLM_MIX._call("deepseek", "q", "be nice", 0.5)
        msgs = client.chat.completions.create.call_args.kwargs["messages"]
        assert msgs[0] == {"role": "system", "content": "be nice"}

    def test_empty_reply_fallback(self):
        client = _fake_openai(None)
        with patch.object(LLM_MIX, "OpenAI", return_value=client):
            assert LLM_MIX._call("deepseek", "q", None, 0.5) == "[空回复]"

    def test_kimi_temperature_quirk(self):
        """⚠️ 注意: 代码里 `temperature != True` 实际是在和 1 比较（True==1）。
        结果 kimi 永远被强制成 temperature=1，传任何值都没用。
        这个测试记录的是当前行为——你需要确认这是不是你想要的。"""
        client = _fake_openai()
        with patch.object(LLM_MIX, "OpenAI", return_value=client):
            LLM_MIX._call("kimi", "q", None, 0.2)
        assert client.chat.completions.create.call_args.kwargs["temperature"] == 1


# ===========================================================================
# 4. MCP 工具集成测试（走 FastMCP 内存客户端）
# ===========================================================================
class TestHealthAndSessions:
    # 这一组走真正的 FastMCP tool 调用路径,但不触发真实 API。
    # 用来确认工具已注册、返回结构符合预期、全局 session 管理能工作。
    async def test_health_check(self):
        data = await _call_tool("health_check", {})
        assert data["success"] is True
        assert data["mcp_name"] == "LLM_MIX"
        assert set(data["providers"]) == {"deepseek", "kimi", "qwen", "gpt"}

    async def test_list_sessions_empty(self):
        data = await _call_tool("list_sessions", {})
        assert data["count"] == 0

    async def test_clear_session_nonexistent(self):
        data = await _call_tool("clear_session", {"session_id": "no-such-id"})
        assert "不存在" in data

    async def test_clear_all_sessions(self):
        LLM_MIX.SESSIONS["s1"] = [{"role": "user", "content": "x"}]
        LLM_MIX.SESSIONS["s2"] = [{"role": "user", "content": "y"}]
        data = await _call_tool("clear_all_sessions", {})
        assert data["cleared"] == 2
        assert len(LLM_MIX.SESSIONS) == 0



class TestAsk:
    # ask 是有状态会话工具。
    # 这里 mock _call_messages, 只测试 session 创建、续聊、system 消息和失败返回。
    async def test_ask_creates_session(self):
        with patch.object(LLM_MIX, "_call_messages", return_value="MOCK_REPLY"):
            data = await _call_tool("ask", {"prompt": "hi", "model": "deepseek"})
        assert data["success"] is True
        assert data["reply"] == "MOCK_REPLY"
        sid = data["session_id"]
        assert sid in LLM_MIX.SESSIONS
        assert len(LLM_MIX.SESSIONS[sid]) == 2  # user + assistant

    async def test_ask_continues_session(self):
        with patch.object(LLM_MIX, "_call_messages", return_value="R1"):
            d1 = await _call_tool("ask", {"prompt": "Q1", "model": "deepseek"})
        sid = d1["session_id"]
        with patch.object(LLM_MIX, "_call_messages", return_value="R2"):
            d2 = await _call_tool("ask", {"prompt": "Q2", "model": "deepseek",
                                          "session_id": sid})
        assert d2["session_id"] == sid
        assert len(LLM_MIX.SESSIONS[sid]) == 4  # 两轮 user+assistant

    async def test_ask_system_only_first_turn(self):
        with patch.object(LLM_MIX, "_call_messages", return_value="R1"):
            d1 = await _call_tool("ask", {"prompt": "Q1", "model": "deepseek",
                                          "system": "be nice"})
        sid = d1["session_id"]
        assert LLM_MIX.SESSIONS[sid][0]["role"] == "system"
        # 第二轮再传 system 应被忽略，不会重复插入
        with patch.object(LLM_MIX, "_call_messages", return_value="R2"):
            await _call_tool("ask", {"prompt": "Q2", "model": "deepseek",
                                     "session_id": sid, "system": "ignored"})
        sys_msgs = [m for m in LLM_MIX.SESSIONS[sid] if m["role"] == "system"]
        assert len(sys_msgs) == 1

    async def test_ask_api_failure_returns_error(self):
        with patch.object(LLM_MIX, "_call_messages", side_effect=RuntimeError("down")):
            data = await _call_tool("ask", {"prompt": "hi", "model": "deepseek"})
        assert data["success"] is False
        assert "RuntimeError" in data["error"]



class TestAskManyAndReview:
    # ask_many/review 是并行多模型工具。
    # 这里 mock _call, 重点验证单个模型失败不会让整个工具失败。
    async def test_ask_many_all_models(self):
        with patch.object(LLM_MIX, "_call", side_effect=lambda m, *a, **k: f"reply-{m}"):
            data = await _call_tool("ask_many", {"prompt": "hi",
                                                 "models": ["deepseek", "kimi"]})
        assert data["replies"] == {"deepseek": "reply-deepseek", "kimi": "reply-kimi"}

    async def test_ask_many_one_model_fails(self):
        def side(m, *a, **k):
            if m == "kimi":
                raise RuntimeError("boom")
            return f"reply-{m}"
        with patch.object(LLM_MIX, "_call", side_effect=side):
            data = await _call_tool("ask_many", {"prompt": "hi",
                                                 "models": ["deepseek", "kimi"]})
        assert data["replies"]["deepseek"] == "reply-deepseek"
        assert "调用失败" in data["replies"]["kimi"]  # 单个失败不拖垮整体

    async def test_review_aggregates(self):
        with patch.object(LLM_MIX, "_call", side_effect=lambda m, *a, **k: f"opinion-{m}"):
            data = await _call_tool("review", {"content": "some code",
                                               "models": ["deepseek", "gpt"]})
        assert "### deepseek 的评审意见" in data
        assert "### gpt 的评审意见" in data
