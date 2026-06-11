"""
=============================================================================
ReAct Agent Web UI — 基于 Gradio 的流式思考链可视化界面
=============================================================================
同学 A 提供 Agent 核心 While 循环 → 本文件提供 Web 可视化

启动方式：
    pip install gradio requests beautifulsoup4
    python web_ui.py

然后在浏览器打开 http://127.0.0.1:7860
=============================================================================
"""

import re
import json
import requests
import os
import math
import time
import html
from typing import List, Dict, Generator, Optional, Tuple
import urllib.parse
from bs4 import BeautifulSoup

import gradio as gr

# ==========================================
# 模块 1：System Prompt（同学 B 提供）
# ==========================================
SYSTEM_PROMPT = """你是一个严格遵循 ReAct 范式的智能体，使用 Thought → Action → Action Input → Observation 循环解决用户问题，最终以 Final Answer 结束。

# ==================== 铁律 ====================
1. 每一步只能输出一个「Thought → Action → Action Input」组合，或输出 Final Answer 直接结束
2. 每个字段独占一行，字段名后跟英文冒号和空格
3. Action Input 必须是合法的 JSON 字符串，能被 json.loads 解析
4. 不得捏造 Observation，你只输出到 Action Input 为止
5. 不得跳过 Thought，即使很简单也必须写推理过程
6. 信息足够时必须输出 Final Answer: 而非继续调工具

# ==================== 可用工具 ====================
| 工具名     | 功能           | Action Input 格式                     |
|------------|----------------|---------------------------------------|
| calculator | 数学运算       | {"expression": "算式字符串"}           |
| web_search | 搜索互联网     | {"query": "搜索关键词"}                |
| read_file  | 读取文件内容   | {"file_path": "路径", "encoding": "utf-8"} |
| write_file | 写入文件内容   | {"file_path": "路径", "content": "文本内容"} |

# ==================== 输出模板 ====================
Thought: <推理过程>
Action: <工具名>
Action Input: <JSON字符串>

或者结束：

Thought: <总结推理>
Final Answer: <最终回答>

# ==================== Few-Shot 示例 ====================

--- 示例 1：计算器 ---
User: 计算 123 乘以 456

Thought: 用户要求乘法运算，调用 calculator。
Action: calculator
Action Input: {"expression": "123 * 456"}

Observation: 56088

Thought: 结果 56088，可以回答了。
Final Answer: 123 × 456 = 56088

--- 示例 2：多步搜索+计算 ---
User: 上海人口密度是北京的多少倍？

Thought: 需要先查两地的人口和面积数据。先搜上海。
Action: web_search
Action Input: {"query": "上海 人口 面积"}

Observation: 上海人口2487万，面积6340km²。

Thought: 拿到上海数据。接着搜北京。
Action: web_search
Action Input: {"query": "北京 人口 面积"}

Observation: 北京人口2188万，面积16410km²。

Thought: 上海密度=2487/6340≈0.3923，北京密度=2188/16410≈0.1333。用calculator精确计算比值。
Action: calculator
Action Input: {"expression": "(2487 / 6340) / (2188 / 16410)"}

Observation: 2.9417

Thought: 精确比值约2.94，信息充分。
Final Answer: 上海人口密度约为北京的2.94倍。

# ==================== 严禁事项 ====================
- 禁止在 Action Input 中写非 JSON 内容
- 禁止一次输出多个 Action
- 禁止跳过 Thought 直接输出 Action
- 禁止在 Observation 到来前输出 Final Answer
- 禁止用中文冒号（：）替代英文冒号（:）
"""


# ==========================================
# 模块 2：上下文滑动窗口
# ==========================================
def sliding_window_manager(messages: List[Dict[str, str]], max_tokens: int = 4000) -> List[Dict[str, str]]:
    """超限时从旧端裁剪，被裁消息压缩为摘要"""

    def est(text):
        cn = len(re.findall(r'[一-龥]', text))
        en = len(re.findall(r'[a-zA-Z]+', text))
        return int(cn * 1.5 + en * 1.3 + len(text) * 0.2)

    if len(messages) <= 3:
        return messages

    sys_msg = messages[0] if messages[0]["role"] == "system" else None
    rest = messages[1:] if sys_msg else list(messages)

    if sum(est(m["content"]) for m in messages) <= max_tokens:
        return messages

    tail = rest[-1]
    middle = rest[:-1]
    removed = []

    while middle:
        test = ([sys_msg] if sys_msg else []) + middle + [tail]
        if sum(est(m["content"]) for m in test) <= max_tokens:
            break
        removed.append(middle.pop(0))

    result = [sys_msg] if sys_msg else []
    if removed:
        summary = ["[历史摘要]"]
        for m in removed:
            c = m.get("content", "")
            if "Observation:" in c:
                summary.append("工具结果: " + c.split("Observation:")[-1].strip()[:200])
            elif "Final Answer:" in c:
                summary.append("曾得出结论: " + c.split("Final Answer:")[-1].strip()[:200])
        result.append({"role": "user", "content": "\n".join(summary) + "\n\n（以上为历史摘要，请继续推理）"})

    return result + middle + [tail]


# ==========================================
# 模块 3：工具箱（同学 C 提供）
# ==========================================
def calculator(expression: str) -> str:
    """安全计算器"""
    try:
        safe = {'abs': abs, 'round': round, 'sqrt': math.sqrt, 'pow': math.pow,
                'pi': math.pi, 'e': math.e}
        code = compile(expression, '<string>', 'eval')
        for name in code.co_names:
            if name not in safe and name not in dir(math):
                raise NameError(f"禁止使用: {name}")
        result = eval(code, {"__builtins__": {}}, safe)
        if isinstance(result, float):
            if result.is_integer():
                return str(int(result))
            return f"{result:.10f}".rstrip('0').rstrip('.')
        return str(result)
    except ZeroDivisionError:
        return "Error: 除零错误"
    except Exception as e:
        return f"Error: 计算失败 - {e}"


def web_search(query: str) -> str:
    """基于百度搜索的本地化查询工具"""
    if not query or not query.strip():
        return "Error: 查询词为空"

    try:
        encoded_query = urllib.parse.quote(query.strip())
        url = f"https://www.baidu.com/s?wd={encoded_query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'html.parser')
        results = []
        for item in soup.find_all('div', class_='c-container'):
            text = item.get_text(separator=' ', strip=True)
            if text:
                results.append(text)
        if results:
            return "\n---\n".join(results[:3])[:800]
        else:
            return f"未找到关于 '{query}' 的有效信息，或触发了百度的反爬虫机制。"
    except requests.Timeout:
        return "Error: 百度搜索网络超时"
    except Exception as e:
        return f"Error: 百度搜索执行失败 - {e}"


def read_file(file_path: str) -> str:
    """安全的文件读取工具"""
    try:
        if not file_path.lower().endswith('.txt'):
            return "Error: 只支持读取 .txt 文件"
        if ".." in file_path:
            return "Error: 路径遍历攻击检测"
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read(2000)
    except FileNotFoundError:
        return f"Error: 文件 '{file_path}' 不存在"
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="gbk") as f:
                return f.read(2000)
        except Exception:
            return "Error: 无法解码文件"
    except Exception as e:
        return f"Error: 文件读取失败 - {e}"


def write_file(param: str) -> str:
    """安全的文件写入工具"""
    try:
        if ":" not in param:
            return "Error: 写入格式错误，应为 '文件名: 内容'"
        idx = param.find(":")
        path = param[:idx].strip()
        content = param[idx + 1:].strip()
        if not path.lower().endswith('.txt'):
            return "Error: 只支持写入 .txt 文件"
        if ".." in path:
            return "Error: 路径遍历攻击检测"
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content[:1000])
        return "成功写入 " + path
    except Exception as e:
        return f"Error: 文件写入失败 - {e}"


TOOL_MAP = {
    "calculator": calculator,
    "web_search": web_search,
    "read_file": read_file,
    "write_file": write_file,
}


def call_tool(action_name: str, action_input_raw: str) -> str:
    """工具路由分配器"""
    name = action_name.strip()
    if name not in TOOL_MAP:
        return f"Error: 未知工具 '{name}'。可用工具: {list(TOOL_MAP.keys())}"
    try:
        params = json.loads(action_input_raw)
    except json.JSONDecodeError as e:
        return f"Error: 无效的JSON格式 - {e}"

    try:
        if name == "calculator":
            if not params.get("expression"):
                return "Error: 缺少 'expression' 参数"
            return calculator(params["expression"])
        elif name == "web_search":
            if not params.get("query"):
                return "Error: 缺少 'query' 参数"
            return web_search(params["query"])
        elif name == "read_file":
            if not params.get("file_path"):
                return "Error: 缺少 'file_path' 参数"
            return read_file(params["file_path"])
        elif name == "write_file":
            if not params.get("file_path") or not params.get("content"):
                return "Error: 缺少 'file_path' 或 'content' 参数"
            return write_file(f"{params['file_path']}: {params['content']}")
        else:
            return f"Error: 工具 '{name}' 未实现"
    except Exception as e:
        return f"Error: 调用工具 '{name}' 时发生异常 - {e}"


# ==========================================
# 模块 4：流式 ReAct Agent（同学 A 的 run 方法 → 改造为 run_stream 生成器）
# ==========================================
class StreamingReActAgent:
    """
    流式 ReAct Agent
    将原始 run() 方法改造为生成器，每完成一个 Thought→Action→Observation 周期就 yield 一次
    """

    def __init__(self, api_key: str, base_url: str, model_name: str, max_loops: int = 10):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.max_loops = max_loops

        # 匹配 Action + Action Input
        self.action_pattern = re.compile(
            r"Action:\s*(.*?)\s*\n.*?Action Input:\s*(.*)",
            re.IGNORECASE | re.DOTALL
        )
        # 匹配 Thought
        self.thought_pattern = re.compile(
            r"Thought:\s*([\s\S]*?)(?=\n(?:Action|Final Answer):|$)",
            re.IGNORECASE
        )

    def _call_llm(self, messages: List[Dict[str, str]]) -> Tuple[str, Optional[str]]:
        """
        调用大模型 API
        返回 (response_text, error_message)
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1
        }
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'], None
        except requests.Timeout:
            return "", "API 请求超时（60秒），请检查网络连接或 API 服务状态"
        except requests.HTTPError as e:
            status = e.response.status_code if e.response else "未知"
            detail = ""
            if e.response and e.response.text:
                try:
                    detail = e.response.json()
                except Exception:
                    detail = e.response.text[:500]
            return "", f"API HTTP 错误 [{status}]: {detail}"
        except Exception as e:
            return "", f"API 请求失败: {str(e)}"

    def _parse_step(self, response_text: str) -> dict:
        """
        解析 LLM 响应中的 Thought、Action、Action Input、Final Answer
        返回结构化字典
        """
        result = {
            "raw": response_text,
            "thought": "",
            "action": None,
            "action_input": None,
            "final_answer": None,
            "is_final": False,
        }

        # 提取 Thought
        thought_match = self.thought_pattern.search(response_text)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        # 检查是否是 Final Answer
        if "Final Answer:" in response_text:
            result["is_final"] = True
            result["final_answer"] = response_text.split("Final Answer:", 1)[-1].strip()
            return result

        # 提取 Action 和 Action Input
        action_match = self.action_pattern.search(response_text)
        if action_match:
            result["action"] = action_match.group(1).strip()
            result["action_input"] = action_match.group(2).strip()

        return result

    def run_stream(self, user_query: str) -> Generator[dict, None, None]:
        """
        流式运行 ReAct 循环

        每次 yield 返回:
        {
            "step": int,           # 当前步数
            "max_steps": int,      # 最大步数
            "thought": str,        # 思考内容
            "action": str|None,    # 工具名称
            "action_input": str|None,  # 工具参数
            "observation": str|None,   # 工具返回结果
            "final_answer": str|None,  # 最终答案（结束时）
            "error": str|None,     # 错误信息
            "status": str,         # "thinking" | "acting" | "observing" | "done" | "error"
            "raw_llm_output": str, # LLM 原始输出
        }
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]

        for step in range(self.max_loops):
            # --- 阶段 1: 调用 LLM ---
            messages = sliding_window_manager(messages)

            response_text, api_error = self._call_llm(messages)
            if api_error:
                yield {
                    "step": step + 1,
                    "max_steps": self.max_loops,
                    "thought": "",
                    "action": None,
                    "action_input": None,
                    "observation": None,
                    "final_answer": None,
                    "error": api_error,
                    "status": "error",
                    "raw_llm_output": "",
                }
                return

            messages.append({"role": "assistant", "content": response_text})
            parsed = self._parse_step(response_text)

            # --- 阶段 2: 如果是 Final Answer，直接结束 ---
            if parsed["is_final"]:
                yield {
                    "step": step + 1,
                    "max_steps": self.max_loops,
                    "thought": parsed["thought"],
                    "action": None,
                    "action_input": None,
                    "observation": None,
                    "final_answer": parsed["final_answer"],
                    "error": None,
                    "status": "done",
                    "raw_llm_output": response_text,
                }
                return

            # --- 阶段 3: 执行工具调用 ---
            if parsed["action"] is None:
                observation = "System Error: 未匹配到正确的 Action 和 Action Input 格式。请按规范输出，或给出 Final Answer:"
                yield {
                    "step": step + 1,
                    "max_steps": self.max_loops,
                    "thought": parsed["thought"],
                    "action": "(解析失败)",
                    "action_input": None,
                    "observation": observation,
                    "final_answer": None,
                    "error": "格式解析失败",
                    "status": "error",
                    "raw_llm_output": response_text,
                }
                messages.append({"role": "user", "content": f"Observation: {observation}"})
                continue

            observation = call_tool(parsed["action"], parsed["action_input"])

            # --- 阶段 4: yield 当前步骤的完整信息 ---
            yield {
                "step": step + 1,
                "max_steps": self.max_loops,
                "thought": parsed["thought"],
                "action": parsed["action"],
                "action_input": parsed["action_input"],
                "observation": observation,
                "final_answer": None,
                "error": None,
                "status": "acting",
                "raw_llm_output": response_text,
            }

            messages.append({"role": "user", "content": f"Observation: {observation}"})

        # 达到最大循环次数
        yield {
            "step": self.max_loops,
            "max_steps": self.max_loops,
            "thought": "",
            "action": None,
            "action_input": None,
            "observation": None,
            "final_answer": "❌ 任务失败：达到最大循环次数限制。",
            "error": "达到最大循环次数限制",
            "status": "error",
            "raw_llm_output": "",
        }


# ==========================================
# 模块 5：Gradio Web UI — 漂亮的流式界面
# ==========================================

# --- 自定义 CSS ---
CUSTOM_CSS = """
/* ========== 全局 ========== */
.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
}

/* ========== 标题 ========== */
.main-title {
    text-align: center;
    font-size: 2.2em !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px !important;
}
.subtitle {
    text-align: center;
    color: #888;
    font-size: 0.9em;
    margin-top: 0;
    margin-bottom: 16px;
}

/* ========== 思考链容器 ========== */
#thought-chain-container {
    min-height: 60px;
    max-height: 500px;
    overflow-y: auto;
    padding: 4px;
}

/* ========== 步骤卡片 ========== */
.step-card {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 12px;
    margin-bottom: 12px;
    overflow: hidden;
    animation: slideIn 0.3s ease-out;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}
.step-header {
    background: linear-gradient(135deg, #45475a 0%, #313244 100%);
    padding: 8px 16px;
    font-size: 0.85em;
    font-weight: 600;
    color: #cdd6f4;
    display: flex;
    align-items: center;
    gap: 8px;
}
.step-badge {
    background: #cba6f7;
    color: #1e1e2e;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 700;
}
.step-body {
    padding: 14px 16px;
}

/* ========== 思考链各区块 ========== */
.thought-block {
    background: #1e2240;
    border-left: 3px solid #89b4fa;
    padding: 10px 14px;
    margin-bottom: 8px;
    border-radius: 0 8px 8px 0;
    font-size: 0.92em;
    color: #ffffff !important;
    line-height: 1.6;
}
.thought-block .label {
    color: #ffffff !important;
    font-weight: 700;
    font-size: 0.82em;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.thought-block div {
    color: #ffffff !important;
}

.action-block {
    background: #1e1e1e;
    border-left: 3px solid #fab387;
    padding: 10px 14px;
    margin-bottom: 8px;
    border-radius: 0 8px 8px 0;
    color: #ffffff !important;
}
.action-block .label {
    color: #ffffff !important;
    font-weight: 600;
    font-size: 0.8em;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.action-block div {
    color: #ffffff !important;
}
.action-block pre {
    background: #11111b;
    color: #ffffff !important;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 0.85em;
    margin: 6px 0 0 0;
    overflow-x: auto;
    font-family: 'Cascadia Code', 'Fira Code', 'JetBrains Mono', monospace;
}

.obs-block {
    background: #1e3020;
    border-left: 3px solid #6fcf6f;
    padding: 10px 14px;
    border-radius: 0 8px 8px 0;
    color: #ffffff !important;
    font-size: 0.9em;
    line-height: 1.5;
    word-break: break-all;
}
.obs-block .label {
    color: #ffffff !important;
    font-weight: 700;
    font-size: 0.82em;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.obs-block div {
    color: #ffffff !important;
}

/* ========== 最终答案卡片 ========== */
.final-answer-card {
    background: linear-gradient(135deg, #1a2e1a 0%, #1e3a1e 100%);
    border: 2px solid #a6e3a1;
    border-radius: 14px;
    padding: 20px 24px;
    margin-top: 12px;
    animation: fadeInUp 0.4s ease-out;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.final-answer-card .fa-header {
    color: #a6e3a1;
    font-size: 1.1em;
    font-weight: 700;
    margin-bottom: 10px;
}
.final-answer-card .fa-content {
    color: #cdd6f4;
    font-size: 1.05em;
    line-height: 1.7;
}

/* ========== 错误卡片 ========== */
.error-card {
    background: #2e1a1a;
    border: 2px solid #f38ba8;
    border-radius: 14px;
    padding: 16px 20px;
    margin-top: 12px;
}
.error-card .err-header {
    color: #f38ba8;
    font-weight: 700;
    font-size: 1em;
}
.error-card .err-content {
    color: #f5c2e7;
    margin-top: 6px;
    font-size: 0.9em;
}

/* ========== 状态栏 ========== */
.status-bar {
    text-align: center;
    padding: 8px;
    font-size: 0.85em;
    color: #6c7086;
}
.status-bar.running {
    color: #f9e2af;
}
.status-bar.done {
    color: #a6e3a1;
}
.status-bar.error {
    color: #f38ba8;
}

/* ========== 日志区域滚动条美化 ========== */
#thought-chain-container::-webkit-scrollbar {
    width: 6px;
}
#thought-chain-container::-webkit-scrollbar-track {
    background: transparent;
}
#thought-chain-container::-webkit-scrollbar-thumb {
    background: #45475a;
    border-radius: 3px;
}
"""

# --- 构建思考链 HTML ---
ICON_THOUGHT = "💭"
ICON_ACTION = "⚡"
ICON_OBS = "🔍"
ICON_DONE = "✅"
ICON_ERROR = "❌"


def format_thought_chain_html(steps: list, is_running: bool = False) -> str:
    """将步骤列表渲染为 HTML"""
    if not steps:
        if is_running:
            return """
            <div style="text-align:center;padding:30px;color:#6c7086;">
                <div style="font-size:2em;margin-bottom:10px;">🤔</div>
                <div>Agent 正在思考中...</div>
                <div style="margin-top:8px;font-size:0.8em;">首次调用大模型，请稍候</div>
            </div>
            """
        return """
        <div style="text-align:center;padding:20px;color:#585b70;">
            <div style="font-size:1.5em;">📭</div>
            <div>等待你的提问...</div>
        </div>
        """

    html_parts = []
    for i, step in enumerate(steps):
        status = step.get("status", "acting")
        step_num = step.get("step", i + 1)
        thought = step.get("thought", "")
        action = step.get("action", "")
        action_input = step.get("action_input", "")
        observation = step.get("observation", "")
        error = step.get("error", "")

        # 步骤卡片头部
        if status == "done":
            badge_class = "step-badge"
            badge_text = f"✅ 第 {step_num} 步 · 完成"
            header_style = "background: linear-gradient(135deg, #2a4a2a 0%, #1e3a1e 100%);"
        elif status == "error":
            badge_class = "step-badge"
            badge_text = f"❌ 第 {step_num} 步 · 出错"
            header_style = "background: linear-gradient(135deg, #4a2a2a 0%, #3a1e1e 100%);"
        else:
            badge_class = "step-badge"
            badge_text = f"🔄 第 {step_num} 步"
            header_style = ""

        html_parts.append(f"""
        <div class="step-card">
            <div class="step-header" style="{header_style}">
                <span class="{badge_class}">{badge_text}</span>
            </div>
            <div class="step-body">
        """)

        # Thought 区块
        if thought:
            thought_escaped = html.escape(thought)
            html_parts.append(f"""
                <div class="thought-block">
                    <div class="label">{ICON_THOUGHT} Thought</div>
                    <div>{thought_escaped}</div>
                </div>
            """)

        # Action 区块
        if action:
            action_escaped = html.escape(action)
            html_parts.append(f"""
                <div class="action-block">
                    <div class="label">{ICON_ACTION} Action: {action_escaped}</div>
            """)
            if action_input:
                input_escaped = html.escape(action_input)
                html_parts.append(f'<pre>{input_escaped}</pre>')
            html_parts.append('</div>')

        # Observation 区块
        if observation:
            obs_escaped = html.escape(observation)
            html_parts.append(f"""
                <div class="obs-block">
                    <div class="label">{ICON_OBS} Observation</div>
                    <div>{obs_escaped}</div>
                </div>
            """)

        # 错误信息
        if error and status != "done":
            html_parts.append(f"""
                <div class="error-card" style="margin-top:8px;">
                    <div class="err-header">⚠️ 注意</div>
                    <div class="err-content">{html.escape(error)}</div>
                </div>
            """)

        html_parts.append("</div></div>")  # close step-body and step-card

    # 如果正在运行中，添加加载动画
    if is_running and steps and steps[-1].get("status") not in ("done", "error"):
        html_parts.append("""
        <div style="text-align:center;padding:12px;color:#f9e2af;">
            <span style="display:inline-block;animation:pulse 1.5s ease-in-out infinite;">
                ⏳ Agent 继续思考中...
            </span>
        </div>
        <style>
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        </style>
        """)

    return "\n".join(html_parts)


def format_final_answer_html(answer: str) -> str:
    """格式化最终答案 HTML"""
    if not answer:
        return ""
    answer_escaped = html.escape(answer)
    return f"""
    <div class="final-answer-card">
        <div class="fa-header">{ICON_DONE} 最终答案</div>
        <div class="fa-content">{answer_escaped}</div>
    </div>
    """


def format_error_html(error: str) -> str:
    """格式化错误信息 HTML"""
    return f"""
    <div class="error-card">
        <div class="err-header">{ICON_ERROR} 执行出错</div>
        <div class="err-content">{html.escape(error)}</div>
    </div>
    """


# --- Gradio 主回调（生成器，实现流式更新）---
def run_agent_stream(
    user_query: str,
    api_key: str,
    base_url: str,
    model_name: str,
    max_loops: int,
) -> Generator[Tuple[str, str, str, str], None, None]:
    """
    生成器函数：逐步运行 Agent 并 yield (thought_chain_html, final_answer_html, error_html, status_html)
    Gradio 会在每次 yield 时更新 UI
    """
    # 验证输入
    if not user_query or not user_query.strip():
        yield (
            format_thought_chain_html([], is_running=False),
            "",
            "",
            '<div class="status-bar error">⚠️ 请输入你的问题</div>',
        )
        return

    if not api_key or not api_key.strip():
        yield (
            format_thought_chain_html([], is_running=False),
            "",
            format_error_html("请先在左侧设置中填入 API Key"),
            '<div class="status-bar error">❌ 缺少 API Key</div>',
        )
        return

    # 初始化 Agent
    agent = StreamingReActAgent(
        api_key=api_key.strip(),
        base_url=base_url.strip(),
        model_name=model_name.strip(),
        max_loops=max_loops,
    )

    # 初始状态
    yield (
        format_thought_chain_html([], is_running=True),
        "",
        "",
        '<div class="status-bar running">🚀 Agent 启动中，正在调用大模型...</div>',
    )

    # 收集所有步骤
    all_steps = []
    final_answer = ""
    error_msg = ""

    try:
        for step_data in agent.run_stream(user_query.strip()):
            all_steps.append(step_data)

            status = step_data.get("status", "acting")

            if status == "done":
                final_answer = step_data.get("final_answer", "")
                yield (
                    format_thought_chain_html(all_steps, is_running=False),
                    format_final_answer_html(final_answer),
                    "",
                    f'<div class="status-bar done">✅ 完成！共执行 {len(all_steps)} 步推理</div>',
                )
                return

            elif status == "error":
                error_msg = step_data.get("error", "未知错误")
                # 如果已经有最终答案（达到上限），显示它
                if step_data.get("final_answer"):
                    final_answer = step_data["final_answer"]

                yield (
                    format_thought_chain_html(all_steps, is_running=False),
                    format_final_answer_html(final_answer) if final_answer else "",
                    format_error_html(error_msg) if not final_answer else "",
                    f'<div class="status-bar error">❌ {error_msg}</div>',
                )
                return

            else:
                # 中间步骤：更新思考链，继续运行
                yield (
                    format_thought_chain_html(all_steps, is_running=True),
                    "",
                    "",
                    f'<div class="status-bar running">🔄 第 {step_data["step"]}/{max_loops} 步推理中...</div>',
                )

    except Exception as e:
        yield (
            format_thought_chain_html(all_steps, is_running=False),
            "",
            format_error_html(f"发生未预期的异常: {str(e)}"),
            f'<div class="status-bar error">❌ 异常: {html.escape(str(e)[:100])}</div>',
        )


# --- 构建 Gradio Blocks UI ---
def build_ui():
    """构建完整的 Gradio Web 界面"""
    theme = gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(
        title="ReAct Agent - Thinking Chain Visualization",
    ) as demo:

        # ==================== 标题区 ====================
        gr.Markdown(
            """
            <h1 class="main-title">🤖 ReAct Agent 思考链可视化</h1>
            <p class="subtitle">
                Thought → Action → Observation 循环 · 流式实时展示 · DeepSeek / OpenAI 兼容 API
            </p>
            """
        )

        # ==================== 主体布局 ====================
        with gr.Row(equal_height=False):
            # --- 左侧：设置面板 ---
            with gr.Column(scale=2, min_width=260):
                gr.Markdown("### ⚙️ 设置")

                with gr.Group():
                    api_key_input = gr.Textbox(
                        label="🔑 API Key",
                        placeholder="sk-...",
                        type="password",
                        value="sk-cc02e49237a041f78849cce9770ae064",
                        info="支持 DeepSeek、OpenAI 等兼容 API",
                    )
                    base_url_input = gr.Textbox(
                        label="🌐 API Base URL",
                        value="https://api.deepseek.com/v1/chat/completions",
                        info="OpenAI 兼容的 Chat Completions 端点",
                    )
                    model_name_input = gr.Textbox(
                        label="🧠 模型名称",
                        value="deepseek-chat",
                        info="例如 deepseek-chat, gpt-4o, gpt-3.5-turbo",
                    )
                    max_loops_slider = gr.Slider(
                        label="🔄 最大推理步数",
                        minimum=3,
                        maximum=20,
                        value=10,
                        step=1,
                        info="Agent 最多执行多少轮 Thought→Action 循环",
                    )

                # 快速预设
                gr.Markdown("##### 🔖 快捷预设")
                with gr.Row():
                    preset_deepseek = gr.Button("DeepSeek", size="sm", variant="secondary")
                    preset_openai = gr.Button("OpenAI", size="sm", variant="secondary")
                    preset_local = gr.Button("本地 Ollama", size="sm", variant="secondary")

                gr.Markdown("---")
                gr.Markdown("""
                ##### 💡 示例问题
                - 计算 (1234 + 5678) × 3.14 除以 2
                - 上海人口密度是北京的多少倍？
                - 帮我算一下 log2(1024) + sqrt(144)
                """)

            # --- 右侧：主交互区 ---
            with gr.Column(scale=5, min_width=500):
                # 输入区
                gr.Markdown("### 💬 你的问题")
                user_input = gr.Textbox(
                    label="",
                    placeholder="在此输入你的问题，例如：上海人口密度是北京的多少倍？",
                    lines=3,
                    max_lines=6,
                    show_label=False,
                    container=True,
                )

                with gr.Row():
                    send_btn = gr.Button(
                        "🚀 发送给 Agent",
                        variant="primary",
                        size="lg",
                        scale=2,
                    )
                    clear_btn = gr.Button(
                        "🗑️ 清空",
                        variant="secondary",
                        size="lg",
                        scale=1,
                    )

                # 状态栏
                status_display = gr.HTML(
                    value='<div class="status-bar">等待你的提问...</div>',
                    elem_id="status-display",
                )

                # 思考链折叠面板
                with gr.Accordion(
                    label="🧠 思考链（Thought → Action → Observation）",
                    open=True,
                    elem_id="thought-accordion",
                ):
                    thought_chain_display = gr.HTML(
                        value=format_thought_chain_html([]),
                        elem_id="thought-chain-container",
                    )

                # 最终答案区
                gr.Markdown("### 📝 最终答案")
                final_answer_display = gr.HTML(
                    value="",
                    elem_id="final-answer-display",
                )
                error_display = gr.HTML(
                    value="",
                    elem_id="error-display",
                )

        # ==================== 底部信息栏 ====================
        gr.Markdown("""
        ---
        <div style="text-align:center;color:#6c7086;font-size:0.8em;">
            Built with Gradio · ReAct Agent · DeepSeek / OpenAI Compatible API · Streaming Thought Chain Visualization
        </div>
        """)

        # ==================== 事件绑定 ====================

        # 发送按钮
        send_event = send_btn.click(
            fn=run_agent_stream,
            inputs=[
                user_input,
                api_key_input,
                base_url_input,
                model_name_input,
                max_loops_slider,
            ],
            outputs=[
                thought_chain_display,
                final_answer_display,
                error_display,
                status_display,
            ],
        )

        # 回车提交（在输入框中按 Ctrl+Enter 发送）
        user_input.submit(
            fn=run_agent_stream,
            inputs=[
                user_input,
                api_key_input,
                base_url_input,
                model_name_input,
                max_loops_slider,
            ],
            outputs=[
                thought_chain_display,
                final_answer_display,
                error_display,
                status_display,
            ],
        )

        # 清空按钮
        def clear_all():
            return (
                "",                                                    # user_input
                format_thought_chain_html([]),                        # thought_chain
                "",                                                    # final_answer
                "",                                                    # error
                '<div class="status-bar">等待你的提问...</div>',        # status
            )

        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[
                user_input,
                thought_chain_display,
                final_answer_display,
                error_display,
                status_display,
            ],
        )

        # 快捷预设
        def set_preset(preset_name):
            presets = {
                "deepseek": ("sk-cc02e49237a041f78849cce9770ae064",
                             "https://api.deepseek.com/v1/chat/completions",
                             "deepseek-chat"),
                "openai": ("sk-your-openai-key-here",
                           "https://api.openai.com/v1/chat/completions",
                           "gpt-4o"),
                "ollama": ("ollama",
                           "http://localhost:11434/v1/chat/completions",
                           "qwen2.5:7b"),
            }
            p = presets.get(preset_name, presets["deepseek"])
            return p[0], p[1], p[2]

        preset_deepseek.click(
            fn=lambda: set_preset("deepseek"),
            inputs=[],
            outputs=[api_key_input, base_url_input, model_name_input],
        )
        preset_openai.click(
            fn=lambda: set_preset("openai"),
            inputs=[],
            outputs=[api_key_input, base_url_input, model_name_input],
        )
        preset_local.click(
            fn=lambda: set_preset("ollama"),
            inputs=[],
            outputs=[api_key_input, base_url_input, model_name_input],
        )

    return demo


# ==========================================
# 启动入口
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("[ReAct Agent Web UI] Starting...")
    print("=" * 60)
    print()
    print("[Dependency Check]")
    try:
        import gradio
        print(f"   [OK] Gradio {gradio.__version__}")
    except ImportError:
        print("   [ERR] Gradio not installed. Run: pip install gradio")
        exit(1)
    try:
        import bs4
        print(f"   [OK] BeautifulSoup4")
    except ImportError:
        print("   [ERR] BeautifulSoup4 not installed. Run: pip install beautifulsoup4")
        exit(1)
    print(f"   [OK] requests (built-in)")
    print()
    print("Open http://127.0.0.1:7860 in your browser")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)

    ui = build_ui()
    ui.queue(default_concurrency_limit=5).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        favicon_path=None,
        css=CUSTOM_CSS,
    )
