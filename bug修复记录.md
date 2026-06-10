# AI 提示词垃圾桶

## 2026-06-10

### 问题1：工具名称中英文不一致
- **现象**：同学B Prompt 使用英文工具名，我最初使用中文名
- **修复**：统一改为 `calculator`, `web_search`, `read_file`, `write_file`
- **结果**：与 Prompt 完全对齐

### 问题2：JSON 参数解析失败
- **现象**：模型输出 `Action Input: {"expression": "2+3"}`，但 `call_tool` 未解析 JSON
- **修复**：在 `call_tool` 中增加 `json.loads` 解析，并处理异常
- **结果**：正确提取参数

### 问题3：Windows 路径写入被截断
- **现象**：`write_file` 返回 `"成功写入 C"` 而非完整路径
- **修复**：使用 `"成功写入 " + path` 代替 f