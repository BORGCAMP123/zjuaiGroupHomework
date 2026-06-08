'''text
'''mermaid
graph TD
    %% 样式定义
    classDef ui fill:#f9f2f4,stroke:#c7254e,stroke-width:2px;
    classDef core fill:#e8f4f8,stroke:#31708f,stroke-width:2px;
    classDef context fill:#dff0d8,stroke:#3c763d,stroke-width:2px;
    classDef tool fill:#fcf8e3,stroke:#8a6d3b,stroke-width:2px;
    classDef external fill:#eee,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5;

    subgraph "前端交互层 (负责：同学 D)"
        UI[Web UI 可视化界面 <br/>Gradia/Streamlit]:::ui
    end

    subgraph "Agent 核心调度层 (负责：同学 A | 交付：main.py)"
        Agent[NativeReActAgent 类 <br/>(引擎入口)]:::core
        Loop((While 循环状态机 <br/>Max Loops 熔断保护)):::core
        Parser[Regex 正则解析器 <br/>(提取 Action & Input)]:::core
        
        Agent -->|接收 user_query| Loop
        Loop -->|1. 发送对话历史| LLM_API
        LLM_API -->|2. 返回模型输出| Loop
        Loop -->|3. 拦截 Final Answer| Output[输出最终结果给用户]
        Loop -->|3. 拦截 Action 推理| Parser
    end

    subgraph "上下文与算法层 (负责：同学 B | 交付：prompt.py)"
        SysPrompt[get_system_prompt<br/>(ReAct System 提示词与 Few-Shot)]:::context
        Window[sliding_window_manager<br/>(历史记录截断与摘要)]:::context
    end

    subgraph "本地工具与测试层 (负责：同学 C | 交付：tools.py)"
        ToolRouter[call_tool 路由接口 <br/>(异常捕捉与降级)]:::tool
        Calc[计算器工具 <br/>(四则运算)]:::tool
        Search[网络搜索工具 <br/>(信息检索)]:::tool
        FileIO[文件读写工具 <br/>(本地 I/O)]:::tool
        
        ToolRouter --> Calc
        ToolRouter --> Search
        ToolRouter --> FileIO
    end

    subgraph "外部依赖 (External API)"
        LLM_API[大语言模型 API <br/>(DeepSeek / OpenAI 格式)]:::external
    end

    %% 模块间连线
    UI -->|启动任务| Agent
    UI -.->|实时监听 Thought 推理流| Loop
    
    Agent -.->|初始化加载| SysPrompt
    Loop -.->|每次循环前压缩 Token| Window
    
    Parser -->|4. 解析出工具请求| ToolRouter
    ToolRouter -->|5. 返回 Observation 结果拼入历史| Loop