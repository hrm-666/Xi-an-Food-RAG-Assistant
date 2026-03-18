# Task2 RAG Demo

这是一个基于 LangChain 的本地 RAG 问答示例项目。程序会读取 `rag_docs/` 目录中的 PDF 文档，完成文本切分、向量化入库，并通过检索增强生成回答。

项目当前面向命令行交互，适合课程作业演示、RAG 流程练习和本地实验。

## 功能概览

- 自动加载 `rag_docs/` 中的 PDF 文档
- 使用 `RecursiveCharacterTextSplitter` 进行文本切分
- 使用 HuggingFace 中文向量模型构建 Chroma 向量库
- 使用 DeepSeek 兼容接口完成问答生成
- 返回答案时附带参考来源

## 项目结构

```text
.
├─ main.py                # 程序入口
├─ requirements.txt       # Python 依赖
├─ rag_docs/              # 知识库文档目录
├─ vector_db/             # 本地向量数据库目录
└─ utils/
   ├─ loader.py           # 文档加载
   ├─ splitter.py         # 文本切分
   ├─ database.py         # 向量库构建
   └─ qa_chain.py         # 问答链构建
```

## 运行环境

- Python 3.9+
- 建议使用虚拟环境

## 安装依赖

```bash
pip install -r requirements.txt
```

## 环境变量

请在项目根目录创建 `.env` 文件，并配置：

```env
DEEPSEEK_API_KEY=your_api_key_here
```

说明：

- 不要把真实密钥提交到 Git 仓库
- `.env` 已在 `.gitignore` 中排除

## 启动方式

```bash
python main.py
```

启动后可在命令行输入问题，输入 `exit` 或 `quit` 退出程序。

## 工作流程

1. 从 `rag_docs/` 读取 PDF 文档
2. 将文档切分为多个文本块
3. 使用嵌入模型生成向量并写入 Chroma
4. 根据用户问题检索相关内容
5. 将检索结果交给大模型生成答案

## 注意事项

- `vector_db/` 属于本地生成文件，通常不建议提交
- `rag_docs/` 中如果包含受版权、隐私或课程限定资料，请按实际情况决定是否上传
- 首次运行时会下载部分模型依赖，耗时会比后续运行更长

## 后续可扩展方向

- 增加更多文档格式支持，如 Markdown、TXT
- 将命令行交互改为 Web 界面
- 增加可配置的检索策略与参数
- 补充异常处理、日志和测试
