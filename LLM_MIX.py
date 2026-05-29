from pydantic import Field
from openai import OpenAI, RateLimitError, InternalServerError, APIConnectionError
import os
from fastmcp import FastMCP
from typing import Annotated, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures as cf
import uuid
import threading
import subprocess
import shutil
import re
import time,sys
from dataclasses import field,dataclass


SESSIONS: dict[str, list[dict]] = {}
SESSIONS_LOCK = threading.Lock()
SESSION_LOCKS: dict[str, threading.Lock] = {}
MAX_HISTORY_MESSAGES = 20
MAX_OUTPUT_CHARS = 50000
MAX_BUFFER_LINES = 10000
MAX_JOBS = 50


def _append_bounded(buf: list[str], line: str, max_lines: int = MAX_BUFFER_LINES) -> None:
    buf.append(line)
    if len(buf) > max_lines:
        del buf[: max_lines // 5]


def _cleanup_old_jobs() -> None:
    if len(JOBS) <= MAX_JOBS:
        return
    done_jobs = sorted(
        ((j.started_at, jid) for jid, j in JOBS.items()
         if j.status in ("done", "error", "killed", "timeout")),
    )
    to_drop = len(JOBS) - MAX_JOBS
    for _, jid in done_jobs[:to_drop]:
        JOBS.pop(jid, None)

@dataclass
class CodexJob:
      job_id: str
      process:subprocess.Popen
      status: str ="running"
      stdout_buffer: list[str] = field(default_factory=list)
      stderr_buffer: list[str] = field(default_factory=list)
      codex_session_id: str|None = None
      returncode: int | None = None
      started_at: float = field(default_factory=time.time)
      lock:threading.Lock=field(default_factory=threading.Lock)

JOBS:dict[str, CodexJob] = {}
JOBS_LOCK = threading.Lock()


@dataclass
class MultiCallJob:
    job_id: str
    models: list[str]
    futures: dict
    executor: ThreadPoolExecutor
    results: dict[str, str] = field(default_factory=dict)
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)

MULTI_JOBS: dict[str, MultiCallJob] = {}
MULTI_JOBS_LOCK = threading.Lock()

mcp=FastMCP("LLM_MIX")

def _get_session_lock(sid: str) -> threading.Lock:
    with SESSIONS_LOCK:
        lock = SESSION_LOCKS.get(sid)
        if lock is None:
            lock = threading.Lock()
            SESSION_LOCKS[sid] = lock
        return lock
RETRYABLE_ERRORS = (
      RateLimitError,
      InternalServerError,
      APIConnectionError,
  )


PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "api_key": "sk-",
        "default_model": "deepseek-v4-pro",
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "api_key_env": "MOONSHOT_API_KEY",
        "api_key": "sk-",
        "default_model": "kimi-k2.6",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "api_key": "sk-",
        "default_model": "qwen3.6-plus",
    },
    "gpt": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "api_key": "sk-",
        "default_model": "gpt-5.4-mini",
    },
    # "claude": {
    #     "base_url": "https://api.anthropic.com/v1/",
    #     "api_key_env": "ANTHROPIC_API_KEY",
    #     "api_key": "sk-ant-xxxx",
    #     "default_model": "claude-opus-4-7",
    # },
}

ModelName = Literal["deepseek", "kimi", "qwen", "gpt"]


def _call_with_retry(func, *args, max_attempts: int = 3, **kwargs):
    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except RETRYABLE_ERRORS as e:
            if attempt == max_attempts-1:
                raise
            wait = 2 ** attempt
            print(
                f"[retry] {type(e).__name__}: {e} | 第 {attempt + 1}/{max_attempts} 次失败,{wait}s 后重试",
                file=sys.stderr,
            )
            time.sleep(wait)


def _call(
        model: ModelName,
        prompt: str,
        system: str | None,
        temperature: float,
        request_timeout: int = 120,
        max_attempts: int = 3,
):
    if model == "kimi" and temperature!=True:
        temperature=1
    set_model = PROVIDERS[model]
    my_api_key = os.environ.get(set_model["api_key_env"])
    if not my_api_key:
        my_api_key=set_model["api_key"]
        if not my_api_key:
            print(f"没有设置{model}的API key",file = sys.stderr)
    client = OpenAI(base_url=set_model["base_url"], api_key=my_api_key, timeout=request_timeout)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": prompt})
    response=_call_with_retry(
        client.chat.completions.create,
        max_attempts=max_attempts,
        model=set_model["default_model"],
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content or "[空回复]"


def _call_messages(model: ModelName, messages: list[dict], temperature: float) -> str:
    set_model = PROVIDERS[model]
    my_api_key = os.environ.get(set_model["api_key_env"]) or set_model["api_key"]
    if not my_api_key:
        print(f"没有设置{model}的API key",file = sys.stderr)
    client = OpenAI(base_url=set_model["base_url"], api_key=my_api_key, timeout=60)
    response = _call_with_retry(
        client.chat.completions.create,
        model=set_model["default_model"],
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content or "[空回复]"

def _truncate(text, max_chars=MAX_OUTPUT_CHARS):
    if not text or len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n\n...[省略 {len(text) - max_chars} 字符]...\n\n" + text[-half:]


def _check_history(history: list[dict]) -> list[dict]:
    if len(history) <= MAX_HISTORY_MESSAGES:
        return history

    if history and history[0].get("role") == "system":
        return [history[0]] + history[-(MAX_HISTORY_MESSAGES - 1):]

    return history[-MAX_HISTORY_MESSAGES:]

def _extract_codex_session_id(stdout: str) -> str | None:
    reg=r"session id:\s*([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
    m=re.search(reg, stdout)
    if m:
        return m.group(1)
    else:
        return None

def _reader(stream, buffer: list[str], job: CodexJob):
    try:
        for line in stream:
            with job.lock:
                _append_bounded(buffer, line)
                if job.codex_session_id is None:
                    sid = _extract_codex_session_id(line)
                    if sid:
                        job.codex_session_id = sid
    except Exception as e:
        print(f"[reader] job={job.job_id} 异常: {type(e).__name__}: {e}",
              file=sys.stderr)
    finally:
        try:
            stream.close()
        except Exception:
            pass

def _spawn_codex_job(prompt: str, cd: str | None, sandbox: str,
                       session_id: str | None) -> CodexJob | dict:
    codex_path = shutil.which("codex")
    if not codex_path:
        return {"success": False, "error": "未找到 codex CLI"}

    if session_id:
        args = [codex_path, "exec", "--color", "never",
                "--sandbox", sandbox,
                "resume", "--skip-git-repo-check", session_id, "-"]
    else:
        if not cd:
            return {"success": False, "error": "新建会话必须提供 cd 工作目录"}
        if not os.path.isdir(cd):
            return {"success": False, "error": f"工作目录不存在: {cd}"}
        args = [codex_path, "exec", "--color", "never", "-C", cd,
                "--sandbox", sandbox, "--skip-git-repo-check", "-"]

    try:
        proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except Exception as e:
        return {"success": False, "error": f"启动 codex 进程失败: {type(e).__name__}: {e}"}

    try:
        proc.stdin.write(prompt)
        proc.stdin.close()
    except Exception as e:
        try:
            proc.terminate()
        except Exception:
            pass
        return {"success": False, "error": f"写入 prompt 到 codex 失败: {type(e).__name__}: {e}"}

    job_id = str(uuid.uuid4())
    job = CodexJob(job_id=job_id, process=proc)

    try:
        threading.Thread(target=_reader, args=(proc.stdout, job.stdout_buffer, job),
                         daemon=True).start()
        threading.Thread(target=_reader, args=(proc.stderr, job.stderr_buffer, job),
                         daemon=True).start()
    except Exception as e:
        try:
            proc.terminate()
        except Exception:
            pass
        return {"success": False, "error": f"启动读取线程失败: {type(e).__name__}: {e}"}

    with JOBS_LOCK:
        JOBS[job_id] = job
        _cleanup_old_jobs()
    return job

def _build_job_response(job: CodexJob) -> dict:
    with job.lock:
        status = job.status
        rc = job.returncode
        sid = job.codex_session_id
        stdout_text = "".join(job.stdout_buffer)
        stderr_text = "".join(job.stderr_buffer)

    if status == "running":
        return {
            "success": True,
            "status": "running",
            "job_id": job.job_id,
            "session_id": sid,
            "partial_output": _truncate(stdout_text),
            "message": f"任务仍在运行,使用 wait_codex(job_id='{job.job_id}') 继续等待",
        }
    if status == "error":
        stderr_out = _truncate(stderr_text, max_chars=10000) if stderr_text else None
    else:
        stderr_out = None
    return {
        "success": True,
        "status": status,
        "job_id": job.job_id,
        "session_id": sid,
        "output": _truncate(stdout_text),
        "stderr": stderr_out,
        "exit_code": rc,
    }

def _wait_for_completion(job: CodexJob, max_seconds: int, poll_interval: float = 0.5) -> None:
      deadline = time.time() + max_seconds
      while time.time() < deadline:
          rc = job.process.poll()
          if rc is not None:
              with job.lock:
                  job.returncode = rc
                  job.status = "done" if rc == 0 else "error"
              return
          time.sleep(poll_interval)


def _spawn_multi_call(prompt: str, models: list[str], system: str | None,
                      temperature: float) -> MultiCallJob:
    executor = ThreadPoolExecutor(max_workers=max(1, len(models)))
    futures = {
        executor.submit(_call, m, prompt, system, temperature,
                        request_timeout=60, max_attempts=1): m
        for m in models
    }
    job = MultiCallJob(
        job_id=str(uuid.uuid4()),
        models=list(models),
        futures=futures,
        executor=executor,
    )
    with MULTI_JOBS_LOCK:
        MULTI_JOBS[job.job_id] = job
    return job


def _wait_for_multi_completion(job: MultiCallJob, max_seconds: int) -> None:
    deadline = time.time() + max_seconds
    pending = [f for f in job.futures if not f.done()]
    while pending and time.time() < deadline:
        remaining = deadline - time.time()
        done, pending = cf.wait(pending, timeout=remaining,
                                return_when=cf.FIRST_COMPLETED)
        with job.lock:
            for fut in done:
                m = job.futures[fut]
                try:
                    job.results[m] = fut.result()
                except Exception as e:
                    job.results[m] = f"[调用失败: {type(e).__name__}: {e}]"
        pending = list(pending)

    with job.lock:
        if all(f.done() for f in job.futures):
            job.status = "done"
            job.executor.shutdown(wait=False)


def _build_multi_response(job: MultiCallJob) -> dict:
    with job.lock:
        status = job.status
        replies_snapshot = dict(job.results)
        pending_models = [m for f, m in job.futures.items() if not f.done()]

    if status == "running":
        return {
            "success": True,
            "status": "running",
            "job_id": job.job_id,
            "partial_replies": replies_snapshot,
            "pending": pending_models,
            "message": f"还有 {len(pending_models)} 个模型未返回,使用 wait_many(job_id='{job.job_id}') 继续等待",
        }
    return {
        "success": True,
        "status": "done",
        "job_id": job.job_id,
        "replies": replies_snapshot,
    }


# @mcp.tool()
# def ask(prompt:Annotated[str,Field(description="要发送给模型的问题或内容")],
#         model: Annotated[ModelName, Field(description="选择哪个 LLM")] = "deepseek",
#         system: Annotated[str | None,Field(description="可选的 system prompt")]=None,
#         temperature: Annotated[float,Field(ge=0, le=2,description="控制回答随机性，范围 0 到 2,数值越低回答越稳定，数值越高回答越发散")] = 0.7,
#         )->str:
#         return _call(model, prompt, system, temperature)


"""
@brief:问指定的单个 LLM,支持多轮会话。
"""
@mcp.tool()
def ask(prompt:Annotated[str,Field(description="要发送给模型的问题或内容")],
        model: Annotated[ModelName, Field(description="选择哪个 LLM")] = "deepseek",
        session_id: Annotated[str | None, Field(description="会话 ID,传入上次返回的 session_id")] = None,
        system: Annotated[str | None,Field(description="可选的 system prompt,仅在会话首轮生效")]=None,
        temperature: Annotated[float,Field(ge=0, le=2,description="控制回答随机性，范围 0 到 2,数值越低回答越稳定，数值越高回答越发散")] = 0.7,
        )->dict:
        sid = session_id or str(uuid.uuid4())
        sess_lock = _get_session_lock(sid)

        with sess_lock:
            with SESSIONS_LOCK:
                history = list(SESSIONS.get(sid, []))
            if not history and system:
                history.append({"role": "system", "content": system})
            history.append({"role": "user", "content": prompt})

            try:
                reply = _call_messages(model, history, temperature)
            except Exception as e:
                return {
                    "success": False,
                    "session_id": sid,
                    "error": f"{type(e).__name__}: {e}",
                }

            history.append({"role": "assistant", "content": reply})
            history = _check_history(history)
            with SESSIONS_LOCK:
                SESSIONS[sid] = history

        return {"success": True, "reply": reply, "session_id": sid}


# """
# @brief:同一个 prompt 广播给多个 LLM 并行回答,多模型广播是一次性的。但是Claude知道要给出上下文信息，所以效果也是有记忆的
# """
# @mcp.tool()
# def ask_many(
#         prompt: Annotated[str, Field(description="发给所有模型的同一个问题")],
#         models: Annotated[list[ModelName], Field(description="参与回答的模型列表,从 deepseek/kimi/qwen/gpt 中选,建议至少 2 个")],
#         system: Annotated[str | None, Field(description="可选的 system prompt,所有模型共用")] = None,
#         temperature: Annotated[float, Field(ge=0, le=2, description="控制回答随机性,范围 0 到 2.")] = 0.7,
# ) -> dict:
#         results: dict[str, str] = {}
#         with ThreadPoolExecutor(max_workers=max(1, len(models))) as ex:
#             futures = {
#                 ex.submit(_call, m, prompt, system, temperature,request_timeout=90,max_attempts=1): m for m in models
#             }
#             for fut in as_completed(futures):
#                 m = futures[fut]
#                 try:
#                     results[m] = fut.result()
#                 except Exception as e:
#                     results[m] = f"[调用失败: {type(e).__name__}: {e}]"
#         return {"replies": results}


"""
@brief:同一个 prompt 广播给多个 LLM 并行回答。
最多等 wait_seconds 秒,没全部回完就返回 status="running" + job_id,后续用 wait_many。
"""
@mcp.tool()
def ask_many(
        prompt: Annotated[str, Field(description="发给所有模型的同一个问题")],
        models: Annotated[list[ModelName], Field(description="参与回答的模型列表,从 deepseek/kimi/qwen/gpt 中选,建议至少 2 个")],
        system: Annotated[str | None, Field(description="可选的 system prompt,所有模型共用")] = None,
        temperature: Annotated[float, Field(ge=0, le=2, description="控制回答随机性,范围 0 到 2.")] = 0.7,
        wait_seconds: Annotated[int, Field(ge=10, le=65, description="本次最多等多少秒,默认 60。Claude Code 的 MCP 单次调用上限约 70s,不要超过 65,否则会被上层判超时报 'Tool result missing'")] = 60,
) -> dict:
    job = _spawn_multi_call(prompt, models, system, temperature)
    _wait_for_multi_completion(job, max_seconds=wait_seconds)
    return _build_multi_response(job)


"""
@brief:继续等待一个之前 ask_many / review 返回了 status="running" 的任务。
用返回的 job_id 调本工具,最多再等 wait_seconds 秒。可反复调用,直到 status 变为 done。
"""
@mcp.tool()
def wait_many(
        job_id: Annotated[str, Field(description="ask_many 或 review 返回的 job_id")],
        wait_seconds: Annotated[int, Field(ge=10, le=65, description="本次最多等多少秒,默认 60。Claude Code 的 MCP 单次调用上限约 70s,不要超过 65,否则会被上层判超时报 'Tool result missing'")] = 60,
) -> dict:
    with MULTI_JOBS_LOCK:
        job = MULTI_JOBS.get(job_id)
    if job is None:
        return {"success": False, "error": f"未找到 job_id={job_id},可能已被清理或从未存在"}

    if job.status == "done":
        return _build_multi_response(job)

    _wait_for_multi_completion(job, max_seconds=wait_seconds)
    return _build_multi_response(job)


"""
@brief:结束并清除指定会话,释放内存
"""
@mcp.tool()
def clear_session(session_id: Annotated[str, Field(description="要清除的会话 ID")]) -> str:
        with SESSIONS_LOCK:
            existed = session_id in SESSIONS
            SESSIONS.pop(session_id, None)
            SESSION_LOCKS.pop(session_id, None)
        return f"会话 {session_id} 已清除" if existed else f"会话 {session_id} 不存在"

# """
# @brief:让多个 LLM 并行审查同一份内容,返回各家意见的汇总
# """
# @mcp.tool()
# def review(
#         content:Annotated[str,Field(description="需要多方互审的内容,如代码、方案、论文片段")],
#         models: Annotated[list[ModelName],Field(description="参与互审的模型列表，可以询问用户的选择,提供ModelName里面的全部模型")],
#         focus:Annotated[str | None, Field(description="审查重点,如 '找 bug' / '检查逻辑严谨性' / '评估可行性'")] = None,
# )->str:
#         system = "你是一位严谨的评审专家。请指出问题、风险与改进建议,简明扼要,避免空泛的客套。"
#         user_prompt = f"请审查以下内容:\n\n{content}"
#         if focus:
#             user_prompt += f"\n\n审查重点:{focus}"
#         results: dict[str, str] = {}
#         with ThreadPoolExecutor(max_workers=max(1, len(models))) as ex:
#             futures = {
#                 ex.submit(_call, m, user_prompt, system, temperature=0.3,request_timeout=90,max_attempts=1): m for m in models
#             }
#             for fut in as_completed(futures):
#                 m = futures[fut]
#                 try:
#                     results[m] = fut.result()
#                 except Exception as e:
#                     results[m] = f"[调用失败: {type(e).__name__}: {e}]"
#         return "\n\n".join(
#             f"### {m} 的评审意见\n{results[m]}" for m in models if m in results
#             )


"""
@brief:让多个 LLM 并行审查同一份内容,异步模式。
最多等 wait_seconds 秒,没全部回完就返回 status="running" + job_id,后续用 wait_many 接力。
"""
@mcp.tool()
def review(
        content: Annotated[str, Field(description="需要多方互审的内容,如代码、方案、论文片段")],
        models: Annotated[list[ModelName], Field(description="参与互审的模型列表,可以询问用户的选择,提供ModelName里面的全部模型")],
        focus: Annotated[str | None, Field(description="审查重点,如 '找 bug' / '检查逻辑严谨性' / '评估可行性'")] = None,
        wait_seconds: Annotated[int, Field(ge=10, le=65, description="本次最多等多少秒,默认 60。Claude Code 的 MCP 单次调用上限约 70s,不要超过 65,否则会被上层判超时报 'Tool result missing'")] = 60,
) -> dict:
    system = "你是一位严谨的评审专家。请指出问题、风险与改进建议,简明扼要,避免空泛的客套。"
    user_prompt = f"请审查以下内容:\n\n{content}"
    if focus:
        user_prompt += f"\n\n审查重点:{focus}"
    job = _spawn_multi_call(user_prompt, models, system, temperature=0.3)
    _wait_for_multi_completion(job, max_seconds=wait_seconds)
    return _build_multi_response(job)


#
# @mcp.tool()
# def ask_codex(
#         prompt:Annotated[str,Field(description="发给 codex 的任务指令")],
#         cd : Annotated[str | None,Field(description="codex 工作目录的绝对路径,仅新建会话时需要.续聊时忽略,沿用上次会话的目录")] = None,
#         sandbox:Annotated[Literal["read-only", "workspace-write", "danger-full-access"],
#         Field(description="沙箱级别,仅新建会话时生效.续聊沿用上次设置。read-only=只读最安全.workspace-write=允许改,要codex干活就用这个 cd;danger-full-access=完全放权,危险,使用前一定要询问 user")]
#         ="read-only",#LLM你好,要codex干活就把sandbox设置为workspace-write
#         session_id:Annotated[str | None,Field(description="会话 ID。传入上次返回的 session_id 可续聊.不传则开启新会话")]=None,
#         timeout: Annotated[int,Field(ge=10, le=1800,description="codex 执行的超时秒数,默认 600s")]=720,
# )-> dict:
#           codex_path = shutil.which("codex")
#           if not codex_path:
#               return {
#                   "success": False,
#                   "error": "未找到 codex CLI。请确认 OpenAI Codex CLI 已安装",
#               }
#
#           if session_id:
#      #      : args = [codex_path, "exec", "resume", "--skip-git-repo-check", session_id, prompt]
#               args = [codex_path, "exec", "--color", "never", "resume", "--skip-git-repo-check", session_id, "-"]
#           else:
#               if not cd:
#                   return {"success": False, "error": "新建会话必须提供 cd 工作目录"}
#               if not os.path.isdir(cd):
#                   return {"success": False, "error": f"工作目录不存在: {cd}"}
#      #      : args = [codex_path, "exec", "-C", cd, "--sandbox", sandbox, "--skip-git-repo-check", prompt]
#               args = [codex_path, "exec", "--color", "never", "-C", cd, "--sandbox", sandbox, "--skip-git-repo-check", "-"]
#
#           try:
#               result = subprocess.run(
#                   args,
#                   input=prompt,
#                   capture_output=True,
#                   text=True,
#                   timeout=timeout,
#                   encoding="utf-8",
#                   errors="replace",
#               )
#               combined_output = (result.stdout or "") + "\n" + (result.stderr or "")
#               return {
#                   "success": result.returncode == 0,
#                   "output": _truncate(result.stdout),#result.stdout
#                   "session_id": _extract_codex_session_id(combined_output),
#                   "stderr": _truncate(result.stderr)if result.stderr else None,
#                   "exit_code": result.returncode,
#               }
#           except subprocess.TimeoutExpired:
#               return {"success": False, "error": f"codex 调用超时(超过 {timeout} 秒)"}
#           except Exception as e:
#               return {"success": False, "error": f"{type(e).__name__}: {e}"}


"""
@brief:控制 codex 干活,异步模式。先启动任务,本次调用最多等 wait_seconds 秒.
没跑完返回 status="running" 和 job_id,后续用 wait_codex 接力等待.
"""
@mcp.tool()
def ask_codex(
        prompt:Annotated[str,Field(description="发给 codex 的任务指令")],
        sandbox:Annotated[Literal["read-only", "workspace-write", "danger-full-access"],
        Field(description="沙箱级别,必填,每次都必须显式传(包括续聊;不会自动沿用上次设置)。read-only=只读,只查代码用;workspace-write=允许改写文件,让codex干活必用;danger-full-access=完全放权,危险,使用前必须问 user")],
        cd : Annotated[str | None,Field(description="codex 工作目录的绝对路径,仅新建会话时需要.续聊时忽略,沿用上次会话的目录")] = None,
        session_id:Annotated[str | None,Field(description="会话 ID。传入上次返回的 session_id 可续聊.不传则开启新会话")]=None,
        wait_seconds: Annotated[int, Field(ge=10, le=65, description="本次最多等多少秒,默认 60。Claude Code 的 MCP 单次调用上限约 70s,不要超过 65,否则会被上层判超时报 'Tool result missing'。没等到就返回 job_id 让 wait_codex 接力")] = 60,
)-> dict:
    spawn_result = _spawn_codex_job(prompt, cd, sandbox, session_id)
    if isinstance(spawn_result, dict):
        return spawn_result
    job = spawn_result
    _wait_for_completion(job, max_seconds=wait_seconds)
    return _build_job_response(job)

"""
@brief:继续等待一个之前 ask_codex 返回了 status="running" 的任务。
用 ask_codex 返回的 job_id 调本工具,最多再等 wait_seconds 秒。可反复调用,直到 status 变为 done/error。
"""
@mcp.tool()
def wait_codex(
          job_id: Annotated[str, Field(description="ask_codex 返回的 job_id")],
          wait_seconds: Annotated[int, Field(ge=10, le=65, description="本次最多等多少秒,默认 60。Claude Code 的 MCP 单次调用上限约 70s,不要超过 65,否则会被上层判超时报 'Tool result missing'")] = 60,
  ) -> dict:
      with JOBS_LOCK:
          job = JOBS.get(job_id)
      if job is None:
          return {"success": False, "error": f"未找到 job_id={job_id},可能已被清理或从未存在"}

      if job.status in ("done", "error", "killed", "timeout"):
          return _build_job_response(job)

      _wait_for_completion(job, max_seconds=wait_seconds)
      return _build_job_response(job)



"""
@brief:检查 MCP 服务运行环境,不返回任何 API key 内容。
"""
@mcp.tool()
def health_check() -> dict:
          codex_path = shutil.which("codex")
          providers_status = {
              name: {
                  "base_url": cfg["base_url"],
                  "default_model": cfg["default_model"],
                  "api_key_env": cfg["api_key_env"],
                  "env_key_set": bool(os.environ.get(cfg["api_key_env"])), ##goto reslove
                  "fallback_key_set": bool(cfg.get("api_key")),
              }
              for name, cfg in PROVIDERS.items()
          }
          return {
              "success": True,
              "mcp_name": "LLM_MIX",
              "python_version": os.sys.version,
              "cwd": os.getcwd(),
              "codex_found": bool(codex_path),
              "codex_path": codex_path,
              "max_history_messages": MAX_HISTORY_MESSAGES,
              "session_count": len(SESSIONS),
              "providers": providers_status,
          }


"""
@brief:列出当前 MCP 内存中的会话概览,不返回完整对话内容。
"""
@mcp.tool()
def list_sessions() -> dict:
          with SESSIONS_LOCK:
              sessions = []
              for sid, history in SESSIONS.items():
                  last_message = history[-1] if history else {}
                  sessions.append({
                      "session_id": sid,
                      "message_count": len(history),
                      "last_role": last_message.get("role"),
                      "last_preview": (last_message.get("content") or "")[:120],
                  })
          return {
              "success": True,
              "count": len(sessions),
              "sessions": sessions,
          }


"""
@brief:清空当前 MCP 内存中的所有 ask 会话。
"""
@mcp.tool()
def clear_all_sessions() -> dict:
          with SESSIONS_LOCK:
              count = len(SESSIONS)
              SESSIONS.clear()
              SESSION_LOCKS.clear()
          return {
              "success": True,
              "cleared": count,
          }


if __name__ == "__main__":
     mcp.run()
