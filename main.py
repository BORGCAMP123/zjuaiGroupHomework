import re
import json
import requests
from typing import List, Dict


# ==========================================
# 接口区 1：留给 同学 B (算法/提示词) 的接口
# ==========================================
def get_system_prompt() -> str:
    """
    【同学 B 负责开发】
    在此处返回严谨的 ReAct System Prompt，包含 Few-Shot 案例。
    接口要求：必须返回规范的 string。
    """
    # 临时占位符，等同学 B 交付
    return "You are a helpful AI. Please think and act step by step..."


def sliding_window_manager(messages: List[Dict[str, str]], max_tokens: int = 4000) -> List[Dict[str, str]]:
    """
    【同学 B 负责开发】
    当 messages 列表过长时，自动截断或调用轻量模型做摘要。
    """
    # 临时占位符：目前直接返回原列表
    return messages


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