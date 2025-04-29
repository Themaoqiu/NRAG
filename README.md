# NRAG 健康领域 RAG 系统 🏥🔍

> 这是一个专为健康领域设计的轻量级检索增强生成(RAG)系统，支持从PDF文件中快速提取和检索专业医学信息喵！

## 🌟 核心功能
- **智能解析**：自动提取PDF中的医学文献内容
- **语义检索**：精准匹配健康相关问答
- **知识管理**：本地化存储知识向量

## 🛠️ 快速配置

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **修改配置文件**
```python
# test.py
docs = ReadFiles("你的/医学文献").get_content(
    max_token_len=2000,  # 文本块最大长度
    cover_content=200    # 文本块重叠长度
)
```

3. **运行系统**
```bash
python test.py
```

