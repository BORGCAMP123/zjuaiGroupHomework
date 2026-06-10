# tools.py - 最终稳定版（单参数 write_file，支持 JSON 路由）
import math
import requests
import os
import json

def calculator(expression: str) -> str:
    try:
        safe = {'abs': abs, 'round': round, 'sqrt': math.sqrt, 'pow': math.pow, 'pi': math.pi, 'e': math.e}
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
    if not query or not query.strip():
        return "Error: 查询词为空"
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query.strip(), "format": "json", "no_html": 1, "skip_disambig": 1}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        abstract = data.get("AbstractText", "")
        if abstract:
            return abstract[:500]
        topics = data.get("RelatedTopics", [])
        for topic in topics:
            if isinstance(topic, dict) and "Text" in topic:
                return topic["Text"][:500]
        return f"未找到关于 '{query}' 的结果"
    except requests.Timeout:
        return "Error: 网络超时"
    except Exception as e:
        return f"Error: 搜索失败 - {e}"

def read_file(file_path: str) -> str:
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
    """单参数，格式 '文件名: 内容'"""
    try:
        if ":" not in param:
            return "Error: 写入格式错误，应为 '文件名: 内容'"
        idx = param.find(":")
        path = param[:idx].strip()
        content = param[idx+1:].strip()
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
    name = action_name.strip()
    if name not in TOOL_MAP:
        return f"Error: 未知工具 '{name}'。可用工具: {list(TOOL_MAP.keys())}"
    try:
        params = json.loads(action_input_raw)
    except json.JSONDecodeError as e:
        return f"Error: 无效的JSON格式 - {e}"

    try:
        if name == "calculator":
            expr = params.get("expression", "")
            if not expr:
                return "Error: 缺少 'expression' 参数"
            return calculator(expr)
        elif name == "web_search":
            query = params.get("query", "")
            if not query:
                return "Error: 缺少 'query' 参数"
            return web_search(query)
        elif name == "read_file":
            file_path = params.get("file_path", "")
            if not file_path:
                return "Error: 缺少 'file_path' 参数"
            return read_file(file_path)
        elif name == "write_file":
            file_path = params.get("file_path", "")
            content = params.get("content", "")
            if not file_path:
                return "Error: 缺少 'file_path' 参数"
            # 注意：底层 write_file 是单参数，需要构造 "file_path: content"
            param_str = f"{file_path}: {content}"
            return write_file(param_str)
        else:
            return f"Error: 工具 '{name}' 未实现"
    except Exception as e:
        return f"Error: 调用工具 '{name}' 时发生异常 - {e}"