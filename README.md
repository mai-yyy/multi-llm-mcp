# multi-llm-mcp

一个用于 Claude Code 的 MCP 工具，支持通过 MCP 调用 Codex CLI 执行任务，并支持多个模型（GPT、Kimi、DeepSeek、Qwen 等）并行调用。

本项目基于 [FastMCP](https://github.com/jlowin/fastmcp) 开发，主要解决两个问题：

1. 让 Claude Code 可以把任务交给 Codex CLI 执行，例如查看代码、修改文件、重构项目等。
2. 让 Claude Code 可以同时调用多个大模型，从不同模型获得回答，便于对比、补充和交叉参考。

## 功能特点

1. **Codex CLI 异步任务执行**：通过 MCP 工具调用 Codex CLI，并使用 `job_id` + `wait_codex` 的方式等待长任务完成，避免 MCP 客户端单次工具调用超时。
2. **支持 Codex 沙盒模式**：`read-only`、`workspace-write`、`danger-full-access` 三种级别。
3. **单模型会话调用**：调用指定模型，并通过 `session_id` 保持多轮上下文。
4. **多模型并行调用**：把同一个问题同时发送给多个模型（GPT、Kimi、DeepSeek、Qwen、Claude）。
5. **长任务轮询机制**：对耗时较长的 Codex 任务或多模型调用，先返回 `job_id`，后续继续通过 `wait_*` 工具等待结果。
6. **环境检查工具**：`health_check` 用于检查运行环境、Codex CLI 是否可用、各模型 API Key 是否配置等（不返回任何 key 内容）。

## 工具列表

| 工具名 | 作用 |
|--------|------|
| `ask` | 调用单个模型，支持多轮会话 |
| `ask_many` | 同时调用多个模型 |
| `wait_many` | 继续等待多模型并行调用任务 |
| `review` | 让多个模型同时分析同一段内容 |
| `ask_codex` | 调用 Codex CLI 执行任务 |
| `wait_codex` | 继续等待 Codex CLI 任务 |
| `clear_session` | 清除指定会话 |
| `clear_all_sessions` | 清除全部会话 |
| `list_sessions` | 查看当前内存中的会话 |
| `health_check` | 检查 MCP 服务运行状态 |

## 环境要求

1. Python 3.10+
2. FastMCP
3. OpenAI Python SDK
4. 可选：OpenAI Codex CLI

安装依赖：

```
pip install fastmcp openai
```

如果需要使用 `ask_codex`，还需确保本机已安装 Codex CLI，并能在命令行直接运行：

```
codex
```

另外，在用 `ask_codex` 调度 Codex 之前，确保 Codex CLI 已经登录过账号（首次使用前先在命令行运行 `codex` 完成登录），否则任务会因为未认证而失败。

## 模型配置

项目中通过 `PROVIDERS` 配置不同模型服务，例如 DeepSeek、Kimi、Qwen、GPT 等。推荐使用环境变量保存模型密钥：

| 模型 | 环境变量 |
|------|----------|
| DeepSeek | `DEEPSEEK_API_KEY` |
| Kimi（Moonshot） | `MOONSHOT_API_KEY` |
| Qwen（DashScope） | `DASHSCOPE_API_KEY` |
| GPT（OpenAI） | `OPENAI_API_KEY` |
| Claude（Anthropic，可选） | `ANTHROPIC_API_KEY` |

在 Windows 上可以用 `setx` 把这些设为用户级环境变量（设完需重开终端 / 重启 Claude Code 才生效），按你要用的模型设即可：

```
setx OPENAI_API_KEY "your-key"
setx DEEPSEEK_API_KEY "your-key"
```

（`MOONSHOT_API_KEY` / `DASHSCOPE_API_KEY` / `ANTHROPIC_API_KEY` 同理。）

如果只是自己本地快速验证，也可以临时把 key 写到代码里的 `PROVIDERS` 中，方便测试。但不要把包含 key 的代码上传到 GitHub，也不要发给别人。准备开源或分享前，应改成环境变量方式，或确认代码中已经没有真实 key。

### 启用 Claude（可选）

代码里 `PROVIDERS` 的 `claude` 块默认是注释状态。要启用 Claude：

1. 取消注释 `PROVIDERS` 中的 `claude` 块；
2. 把 `"claude"` 加进 `ModelName`；
3. 配置 `ANTHROPIC_API_KEY`。

它走的是 Anthropic 的 OpenAI 兼容端点（`base_url="https://api.anthropic.com/v1/"`），模型名填 Claude 的名称即可（如 `claude-opus-4-7`、`claude-sonnet-4-6`、`claude-haiku-4-5`）。

> 小技巧：服务本身跑在 Claude Code 里，所以你可以反过来调度别的 Claude——比如用更便宜的 Haiku 去跑并行的苦力活，或者要一个干净上下文、不被当前对话带偏的 Claude 来做二次判断。

## MCP 配置示例（Claude Code）

在 Claude Code 中，推荐直接用 `claude mcp add` 命令添加本项目。基本格式：

```
claude mcp add --scope user llm-mix -- python /absolute/path/to/LLM_MIX.py
```

其中：

1. `llm-mix` 是这个 MCP 服务的名字，可以自己改。
2. `--scope user` 表示该 MCP 对当前用户全局可用，不只限于某一个项目。
3. `python` 是 Python 启动命令。
4. `/absolute/path/to/LLM_MIX.py` 替换成你本地 `LLM_MIX.py` 的绝对路径。

Windows 示例：

```
claude mcp add --scope user llm-mix -- python "C:\path\to\LLM_MIX.py"
```

如果上面那条添加后用不了（通常是 `python` 不在 PATH 上，或指向了别的 Python 环境），改用下面这条、写上 Python 解释器的完整路径：

```
claude mcp add --scope user llm-mix -- "C:\path\to\python.exe" "C:\path\to\LLM_MIX.py"
```

添加完成后，可以用下面的命令查看是否添加成功：

```
claude mcp list
```

进入 Claude Code 后，也可以输入 `/mcp` 查看 MCP 服务是否已连接。

## 使用示例

连接 MCP 后，可以直接让 Claude Code 使用这些能力：

```
用 Codex 查看当前项目结构。
```

```
让 Codex 在当前目录创建一个测试文件。
```

```
让 Codex 修改这个 Python 文件，修复明显的异常处理问题。
```

```
把这个问题同时发给 GPT、Kimi、DeepSeek 和 Qwen。
```

```
让多个模型一起分析这个方案有没有明显问题。
```

```
让 DeepSeek 单独解释这段代码。
```

如果任务较长，工具可能会返回：

```json
{
  "success": true,
  "status": "running",
  "job_id": "xxxx",
  "message": "任务仍在运行，使用 wait_codex 继续等待"
}
```

这时继续调用对应的 `wait_codex` 或 `wait_many` 即可。

## Codex 沙盒使用建议

1. 一般查看代码、分析项目时，优先使用 `read-only`。
2. 需要 Codex 修改文件时，使用 `workspace-write`。
3. 不建议随意使用 `danger-full-access`，因为这个模式会给 Codex 更高的本机权限。

## 注意事项

1. `ask_codex` 依赖本机 Codex CLI。
2. 多模型调用是否可用，取决于你是否配置了对应模型服务（可用 `health_check` 查看）。
3. 长任务建议使用 `job_id` + `wait_*` 的方式继续等待。
4. 个人本地调试可以图方便，但公开仓库前要检查是否包含真实密钥、私人路径或其他敏感信息。

## 社区 / 致谢

本项目在 [LINUX DO](https://linux.do) 社区分享与讨论，感谢社区佬友的反馈与建议。

## License

MIT License
