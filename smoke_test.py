import LLM_MIX
import time

TEST1=False  #True False
TEST2=False
TEST3=False
TEST4=False
TEST5=True


if TEST1:
    print("health:", LLM_MIX.health_check())
    print("ask deepseek:", LLM_MIX.ask("只回复 OK", model="deepseek"))
    print("ask gpt:", LLM_MIX.ask("只回复 OK", model="gpt"))
    print("ask kimi:", LLM_MIX.ask("只回复 OK", model="kimi"))
    print("ask qwen:", LLM_MIX.ask("只回复 OK", model="qwen"))

    print("ask_many:", LLM_MIX.ask_many(
        prompt="只回复 OK",
        models=["deepseek", "gpt", "kimi", "qwen"],
    ))

    print("review:", LLM_MIX.review(
        content="def add(a,b): return a+b",
        models=["deepseek", "gpt"],
        focus="只说有没有明显问题",
    ))

    print("codex:", LLM_MIX.ask_codex(
        prompt="只回复 OK，不要执行命令。",
        cd=r"D:\Claude_to_other",
        sandbox="read-only",
        timeout=120,
    ))

if TEST2:
    long_text = "\n".join(f"第{i}行: 普通内容" for i in range(3000))
    marker = "FINAL_MARKER_9f3a7c_end"

    prompt = f"""
    下面是一段很长的文本，请只回复最后的标记，不要解释。

    {long_text}

    最后标记：{marker}
    """

    r = LLM_MIX.ask_codex(
        prompt=prompt,
        cd=r"C:\Users\maiyan\Desktop",
        sandbox="read-only",
        timeout=180,
    )

    print(r["success"])
    print(r["output"][-1000:])
    print("marker_ok:", marker in (r.get("output") or ""))

if TEST3:
    s = "A" * 80000
    out = LLM_MIX._truncate(s)

    print("original_len:", len(s))
    print("returned_len:", len(out))
    print("truncated:", "省略" in out)
    print("head:", out[:100])
    print("tail:", out[-100:])


if TEST4:
    job = LLM_MIX._spawn_codex_job("你好,回复一个字", cd=r"D:\Claude_to_other",
                           sandbox="read-only", session_id=None)
    print("spawned:", job.job_id, "pid:", job.process.pid)

    for i in range(30):
        rc = job.process.poll()
        print(f"[{i}s] poll={rc}  session_id={job.codex_session_id}  "
              f"stdout_lines={len(job.stdout_buffer)}")
        if rc is not None:
            break
        time.sleep(1)

if TEST5:
    from LLM_MIX import _spawn_codex_job, _wait_for_completion, JOBS

    job = _spawn_codex_job("你好,回复一个字", cd=r"D:\Claude_to_other",
                           sandbox="read-only", session_id=None)
    print("spawned:", job.job_id)

    _wait_for_completion(job, max_seconds=30)
    print("status:", job.status, "returncode:", job.returncode)
    print("session_id:", job.codex_session_id)
    print("=== STDOUT ===")
    print("".join(job.stdout_buffer))
    # _wait_for_completion(job, max_seconds=2)  # 故意只等 2 秒
    # print("status:", job.status)  # 应该是 "running"
    # print("poll:", job.process.poll())  # None,还在跑
    # import time
    #
    # time.sleep(15)  # 干等
    # print("poll after 20s:", job.process.poll())  # 现在应该是 0 了
    # print("buffer:", "".join(job.stdout_buffer))  # 内容已经在了(reader 一直在干活)