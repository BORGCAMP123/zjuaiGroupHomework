import re
import json
import requests
from typing import List, Dict


# ==========================================
# 接口区 1：留给 同学 B (算法/提示词) 的接口
# ==========================================
def get_system_prompt() -> str:
      return """你是一个严格遵循 ReAct 范式的智能体，使用 Thought → Action → Action Input → Observation 循环解决用户问题，最终以
  Final Answer 结束。

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
      return messages

  替换成：

  def sliding_window_manager(messages, max_tokens=4000):
      """
      上下文滑动窗口：Token超限时从旧端裁剪，被裁消息压缩为摘要。
      System Prompt永久保留，最新消息永久保留。
      """
      # ── 1. Token 估算（简易但够用） ──
      def estimate(text):
          cn = len(re.findall(r'[\u4e00-\u9fff]', text))
          en = len(re.findall(r'[a-zA-Z]+', text))
          return int(cn * 1.5 + en * 1.3 + len(text) * 0.2)

      # ── 2. 消息太少，不用裁 ──
      if len(messages) <= 3:
          return messages

      # ── 3. 分离 System Prompt ──
      system_msg = messages[0] if messages[0]["role"] == "system" else None
      rest = messages[1:] if system_msg else list(messages)

      total = sum(estimate(m["content"]) for m in messages)
      if total <= max_tokens:
  def sliding_window_manager(messages, max_tokens=4000):
      """
      上下文滑动窗口：Token超限时从旧端裁剪，被裁消息压缩为摘要。
      System Prompt永久保留，最新消息永久保留。
      """
      # ── 1. Token 估算（简易但够用） ──
      def estimate(text):
          cn = len(re.findall(r'[\u4e00-\u9fff]', text))
          en = len(re.findall(r'[a-zA-Z]+', text))
          return int(cn * 1.5 + en * 1.3 + len(text) * 0.2)

      # ── 2. 消息太少，不用裁 ──
      if len(messages) <= 3:
          return messages

      # ── 3. 分离 System Prompt ──
      system_msg = messages[0] if messages[0]["role"] == "system" else None
      rest = messages[1:] if system_msg else list(messages)

      total = sum(estimate(m["content"]) for m in messages)
      if total <= max_tokens:
          return messages  # 没超，直接返回

      # ── 4. 从旧端（头部）逐条移除 ──
      tail = rest[-1]        # 最新消息，永远保留
      middle = rest[:-1]     # 可被裁剪的中间部分
      removed = []

      while middle:
          test_msgs = [system_msg] if system_msg else []
          test_msgs += middle
          test_msgs.append(tail)
          if sum(estimate(m["content"]) for m in test_msgs) <= max_tokens:
              break
          removed.append(middle.pop(0))  # 扔掉最旧的一条

      # ── 5. 生成摘要 ──
      result = [system_msg] if system_msg else []
      if removed:
          parts = ["[历史摘要]"]
          for m in removed:
              content = m.get("content", "")
              if m["role"] == "user" and content.startswith("Observation:"):
                  parts.append("工具结果: " + content[12:].strip()[:200])
              elif m["role"] == "assistant" and "Final Answer:" in content:
                  fa = content.split("Final Answer:")[-1].strip()[:300]
                  parts.append("曾得出结论: " + fa)
          result.append({
              "role": "user",
              "content": "\n".join(parts) + "\n\n（以上为历史摘要，请基于此继续推理）"
          })

      result += middle
      result.append(tail)
      return result


# ==========================================
# 接口区 2：留给 同学 C (工具/QA测试) 的接口
# ==========================================
def call_tool(action_name: str, action_input_raw: str) -> str:
    """
    【同学 C 负责开发】
    接收大模型输出的工具名和参数，路由到具体的本地函数（计算器、搜索、文件读写）。
    接口要求：输入和输出都必须是规范的 string！如果报错也要返回带错误信息的 string。
    """
    # 临时占位符，等同学 C 交付 tools.py 后在这里引入
    # import tools
    # return tools.execute(action_name, action_input_raw)
    return f"Mock Observation: 工具 {action_name} 暂未接入。"


# ==========================================
# 核心架构区：同学 A (你) 负责的 ReAct 循环调度
# ==========================================
class NativeReActAgent:
    def __init__(self, api_key: str, base_url: str, model_name: str, max_loops: int = 10):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.max_loops = max_loops

        # 预编译正则：提取 Action 和 Action Input (允许跨行)
        self.action_pattern = re.compile(
            r"Action:\s*(.*?)\s*\n.*?Action Input:\s*(.*)",
            re.IGNORECASE | re.DOTALL
        )

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """调用大模型 API (可替换为具体的国内/国外大模型 API)"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1  # ReAct 范式建议低温度以保证格式稳定
        }
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"System Error: API 请求失败 -> {str(e)}"

    def run(self, user_query: str) -> str:
        """核心 While 循环状态机"""
        # 1. 接入同学 B 的提示词
        system_prompt = get_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        print(f"🚀 任务开始: {user_query}\n" + "=" * 50)

        for step in range(self.max_loops):
            # 2. 接入同学 B 的上下文管理（滑动窗口截断）
            messages = sliding_window_manager(messages)

            print(f"🔄 [Step {step + 1}/{self.max_loops}] 大模型思考中...")
            response_text = self._call_llm(messages)

            messages.append({"role": "assistant", "content": response_text})
            print(f"🤖 LLM 输出:\n{response_text}\n" + "-" * 30)

            # 3. 拦截检查：是否包含最终答案
            if "Final Answer:" in response_text:
                final_answer = response_text.split("Final Answer:")[-1].strip()
                print(f"✅ 获得最终答案: {final_answer}")
                return final_answer

            # 4. 正则解析 Action 和 Action Input
            match = self.action_pattern.search(response_text)
            if not match:
                observation = "System Error: 未匹配到正确的 Action 和 Action Input 格式。请按规范输出，或给出 Final Answer:"
            else:
                action_name = match.group(1).strip()
                action_input_raw = match.group(2).strip()

                # 5. 接入同学 C 的工具拦截器
                observation = call_tool(action_name, action_input_raw)

            print(f"🔍 Observation (工具结果):\n{observation}\n" + "=" * 50)

            # 6. 把结果拼回历史给模型继续推导
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        return "❌ 任务失败：达到最大循环次数限制。"


# 测试入口
if __name__ == "__main__":
    # 配置 API 密钥 (第三周联调时填入真实 Key)
    agent = NativeReActAgent(
        api_key="sk-xxxxxx",
        base_url="https://api.deepseek.com/v1/chat/completions",  # 以兼容 OpenAI 格式的 API 为例
        model_name="deepseek-chat"
    )
    agent.run("测试用例：计算 123 乘以 456")
