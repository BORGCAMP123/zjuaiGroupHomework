"""
同学 B 交付模块：prompt.py
包含：get_system_prompt() — ReAct 范式系统提示词
      sliding_window_manager() — 上下文滑动窗口管理器
"""
import re


def get_system_prompt() -> str:
    """返回严谨的 ReAct System Prompt，包含 Few-Shot 案例"""
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

--- 示例 3：无需工具直接回答 ---
User: 什么是ReAct范式？

Thought: 这是知识性问题，不需要工具，直接用已有知识回答。
Final Answer: ReAct是Reasoning+Acting的交替式智能体范式，核心流程为Thought→Action→Observation循环，直至得出Final Answer。

--- 示例 4：工具失败的错误处理 ---
User: 今天天气怎么样？

Thought: 需要搜索天气。
Action: web_search
Action Input: {"query": "今日天气"}

Observation: 网络超时，搜索失败

Thought: 搜索失败，不能捏造天气。诚实告知用户。
Final Answer: 抱歉，无法获取天气数据（网络超时）。建议打开手机天气App查看，或告诉我你的城市我再试一次。

# ==================== 严禁事项 ====================
- 禁止在 Action Input 中写非 JSON 内容
- 禁止一次输出多个 Action
- 禁止跳过 Thought 直接输出 Action
- 禁止在 Observation 到来前输出 Final Answer
- 禁止用中文冒号（：）替代英文冒号（:）
"""


def sliding_window_manager(messages, max_tokens=4000):
    """上下文滑动窗口：超限时从旧端裁剪，被裁消息压缩为摘要"""

    # Token 估算（中文字符 ×1.5、英文单词 ×1.3、其余字符 ×0.2）
    def est(text):
        cn = len(re.findall(r'[一-鿿]', text))
        en = len(re.findall(r'[a-zA-Z]+', text))
        return int(cn * 1.5 + en * 1.3 + len(text) * 0.2)

    # 消息太少，无需裁剪
    if len(messages) <= 3:
        return messages

    # 分离 system 消息和其余消息
    sys_msg = messages[0] if messages[0]["role"] == "system" else None
    rest = messages[1:] if sys_msg else list(messages)

    # Token 总量未超限，直接返回
    if sum(est(m["content"]) for m in messages) <= max_tokens:
        return messages

    # 从旧端裁剪中间消息，保留最新一条
    tail = rest[-1]
    middle = rest[:-1]
    removed = []

    while middle:
        test = ([sys_msg] if sys_msg else []) + middle + [tail]
        if sum(est(m["content"]) for m in test) <= max_tokens:
            break
        removed.append(middle.pop(0))

    # 组装结果：system + 摘要 + 剩余中间 + 最新消息
    result = [sys_msg] if sys_msg else []
    if removed:
        summary = ["[历史摘要]"]
        for m in removed:
            c = m.get("content", "")
            if "Observation:" in c:
                summary.append("工具结果: " + c.split("Observation:")[-1].strip()[:200])
            elif "Final Answer:" in c:
                summary.append("曾得出结论: " + c.split("Final Answer:")[-1].strip()[:200])
        result.append({
            "role": "user",
            "content": "\n".join(summary) + "\n\n（以上为历史摘要，请继续推理）"
        })

    return result + middle + [tail]
