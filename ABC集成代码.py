import re
import json
import requests
import os
import math
from typing import List, Dict
import urllib.parse
from bs4 import BeautifulSoup

# ==========================================
# 模块 1：同学 B 提供的核心算法与提示词
# ==========================================
def get_system_prompt() -> str:
    """返回严谨的 ReAct System Prompt，包含 Few-Shot 案例（已同步新增 write_file 工具）"""
    return """你是一个严格遵循 ReAct 范式的智能体，使用 Thought → Action → Action Input → Observation 循环解决用户问题，最终以 Final Answer 结束。

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


def sliding_window_manager(messages: List[Dict[str, str]], max_tokens: int = 4000) -> List[Dict[str, str]]:
    """上下文滑动窗口：超限时从旧端裁剪，被裁消息压缩为摘要"""

    def est(text):
        cn = len(re.findall(r'[\u4e00-\u9fa5]', text))
        en = len(re.findall(r'[a-zA-Z]+', text))
        return int(cn * 1.5 + en * 1.3 + len(text) * 0.2)

    if len(messages) <= 3: return messages

    sys_msg = messages[0] if messages[0]["role"] == "system" else None
    rest = messages[1:] if sys_msg else list(messages)

    if sum(est(m["content"]) for m in messages) <= max_tokens: return messages

    tail = rest[-1]
    middle = rest[:-1]
    removed = []

    while middle:
        test = ([sys_msg] if sys_msg else []) + middle + [tail]
        if sum(est(m["content"]) for m in test) <= max_tokens: break
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
# 模块 2：同学 C 提供的工具箱底座
# ==========================================
def calculator(expression: str) -> str:
    try:
        safe = {'abs': abs, 'round': round, 'sqrt': math.sqrt, 'pow': math.pow, 'pi': math.pi, 'e': math.e}
        code = compile(expression, '<string>', 'eval')
        for name in code.co_names:
            if name not in safe and name not in dir(math):
                raise NameError(f"禁止使用: {name}")
        result = eval(code, {"__builtins__": {}}, safe)
        if isinstance(result, float):
            if result.is_integer(): return str(int(result))
            return f"{result:.10f}".rstrip('0').rstrip('.')
        return str(result)
    except ZeroDivisionError:
        return "Error: 除零错误"
    except Exception as e:
        return f"Error: 计算失败 - {e}"


import urllib.parse
from bs4 import BeautifulSoup


def web_search(query: str) -> str:
    """基于百度搜索的本地化查询工具"""
    if not query or not query.strip():
        return "Error: 查询词为空"

    try:
        # 1. 构造百度搜索 URL 并对中文进行 URL 编码
        encoded_query = urllib.parse.quote(query.strip())
        url = f"https://www.baidu.com/s?wd={encoded_query}"

        # 2. 伪装成真实的浏览器（极度重要，否则会被百度直接拦截并返回安全验证页面）
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
        }

        # 3. 发送网络请求
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        resp.encoding = 'utf-8'

        # 4. 使用 BeautifulSoup 暴力解析网页纯文本
        soup = BeautifulSoup(resp.text, 'html.parser')

        # 百度搜索结果通常包裹在 class 包含 'c-container' 的 div 中
        results = []
        for item in soup.find_all('div', class_='c-container'):
            # 提取每个结果块的文本，用空格分隔，去掉多余换行
            text = item.get_text(separator=' ', strip=True)
            if text:
                results.append(text)

        # 5. 拼装并截断返回（防止给大模型塞入太多垃圾信息导致 Token 溢出）
        if results:
            # 只取前 3 条最相关的搜索结果
            combined_text = "\n---\n".join(results[:3])
            # 限制总长度最多 800 字
            return combined_text[:800]
        else:
            return f"未找到关于 '{query}' 的有效信息，或触发了百度的反爬虫机制。"

    except requests.Timeout:
        return "Error: 百度搜索网络超时"
    except Exception as e:
        return f"Error: 百度搜索执行失败 - {e}"


def read_file(file_path: str) -> str:
    try:
        if not file_path.lower().endswith('.txt'): return "Error: 只支持读取 .txt 文件"
        if ".." in file_path: return "Error: 路径遍历攻击检测"
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
    try:
        if ":" not in param: return "Error: 写入格式错误，应为 '文件名: 内容'"
        idx = param.find(":")
        path = param[:idx].strip()
        content = param[idx + 1:].strip()
        if not path.lower().endswith('.txt'): return "Error: 只支持写入 .txt 文件"
        if ".." in path: return "Error: 路径遍历攻击检测"
        d = os.path.dirname(path)
        if d and not os.path.exists(d): os.makedirs(d, exist_ok=True)
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
    """同学 C 的 JSON 路由分配器"""
    name = action_name.strip()
    if name not in TOOL_MAP:
        return f"Error: 未知工具 '{name}'。可用工具: {list(TOOL_MAP.keys())}"
    try:
        params = json.loads(action_input_raw)
    except json.JSONDecodeError as e:
        return f"Error: 无效的JSON格式 - {e}"

    try:
        if name == "calculator":
            if not params.get("expression"): return "Error: 缺少 'expression' 参数"
            return calculator(params["expression"])
        elif name == "web_search":
            if not params.get("query"): return "Error: 缺少 'query' 参数"
            return web_search(params["query"])
        elif name == "read_file":
            if not params.get("file_path"): return "Error: 缺少 'file_path' 参数"
            return read_file(params["file_path"])
        elif name == "write_file":
            if not params.get("file_path") or not params.get("content"): return "Error: 缺少 'file_path' 或 'content' 参数"
            return write_file(f"{params['file_path']}: {params['content']}")
        else:
            return f"Error: 工具 '{name}' 未实现"
    except Exception as e:
        return f"Error: 调用工具 '{name}' 时发生异常 - {e}"


# ==========================================
# 模块 3：同学 A 提供的主循环控制流
# ==========================================
class NativeReActAgent:
    def __init__(self, api_key: str, base_url: str, model_name: str, max_loops: int = 10):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.max_loops = max_loops
        self.action_pattern = re.compile(
            r"Action:\s*(.*?)\s*\n.*?Action Input:\s*(.*)",
            re.IGNORECASE | re.DOTALL
        )

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model_name, "messages": messages, "temperature": 0.1}
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"System Error: API 请求失败 -> {str(e)}"

    def run(self, user_query: str) -> str:
        system_prompt = get_system_prompt()
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]

        print(f"🚀 任务开始: {user_query}\n" + "=" * 50)

        for step in range(self.max_loops):
            messages = sliding_window_manager(messages)
            print(f"🔄 [Step {step + 1}/{self.max_loops}] 大模型思考中...")

            response_text = self._call_llm(messages)
            messages.append({"role": "assistant", "content": response_text})
            print(f"🤖 LLM 输出:\n{response_text}\n" + "-" * 30)

            if "Final Answer:" in response_text:
                final_answer = response_text.split("Final Answer:")[-1].strip()
                print(f"✅ 获得最终答案: {final_answer}")
                return final_answer

            match = self.action_pattern.search(response_text)
            if not match:
                observation = "System Error: 未匹配到正确的 Action 和 Action Input 格式。请按规范输出，或给出 Final Answer:"
            else:
                action_name = match.group(1).strip()
                action_input_raw = match.group(2).strip()
                observation = call_tool(action_name, action_input_raw)

            print(f"🔍 Observation (工具结果):\n{observation}\n" + "=" * 50)
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        return "❌ 任务失败：达到最大循环次数限制。"


# ==========================================
# 对话
# ==========================================
if __name__ == "__main__":
    import os

    # 【可选】如果你有代理软件解决 DuckDuckGo 搜索报错问题，取消注释下面两行并修改端口
    # os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    # os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

    # 1. 填入你真实申请的 API Key
    REAL_API_KEY = "sk-在这里填入你真实的API密钥"

    # 2. 初始化 Agent
    agent = NativeReActAgent(
        api_key="sk-cc02e49237a041f78849cce9770ae064",
        base_url="https://api.deepseek.com/v1/chat/completions",  # 请确保和你的提供商一致
        model_name="deepseek-chat"
    )

    print("=" * 50)
    print("🤖 ReAct Agent 终端已启动！")
    print("💡 你可以问真实的问题，例如：'帮我查一下2024年巴黎奥运会中国队拿了多少金牌，并算一下占总金牌数（329块）的百分比'")
    print("输入 'quit' 或 'exit' 退出程序。")
    print("=" * 50)

    # 3. 开启真实对话循环
    while True:
        user_input = input("\n🧑‍💻 用户: ")

        # 退出指令判断
        if user_input.strip().lower() in ['quit', 'exit']:
            print("👋 Agent 已关闭，再见！")
            break

        # 防止输入回车空字符
        if not user_input.strip():
            continue

        # 将用户的真实输入交给 Agent 运行
        agent.run(user_input)