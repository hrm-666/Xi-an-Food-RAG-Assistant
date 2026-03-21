# Task2 RAG Demo

这是一个基于 LangChain 的本地 RAG 问答示例项目。程序会读取 `rag_docs/` 目录中的 PDF 文档，完成文本切分、向量化入库，并通过检索增强生成回答。

当前项目面向命令行交互，适合课程作业演示、RAG 流程练习和本地实验。项目中的知识库主题是“西安美食”。

## 功能概览

- 自动加载 `rag_docs/` 中的 PDF 文档
- 使用 `RecursiveCharacterTextSplitter` 进行文本切分
- 使用 HuggingFace 中文向量模型构建 Chroma 向量库
- 优先复用已有的本地向量库，避免每次启动都重新嵌入
- 使用 DeepSeek 兼容接口完成问答生成
- 结合 BM25 与向量检索进行混合检索
- 返回答案时附带参考来源

## 项目结构

```text
.
├─ main.py                # 程序入口
├─ requirements.txt       # Python 依赖
├─ README.md              # 项目说明
├─ rag_docs/              # 知识库文档目录
├─ vector_db/             # 本地向量数据库目录
└─ src/
   ├─ loader.py           # 文档加载
   ├─ splitter.py         # 文本切分
   ├─ database.py         # 向量库创建/加载
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
3. 检查 `vector_db/` 中是否已有可复用的 Chroma 向量库
4. 如果已有向量库则直接加载，否则使用嵌入模型生成向量并写入 Chroma
5. 根据用户问题进行混合检索
6. 将检索结果交给大模型生成答案
7. 输出答案并标注参考来源

## 检索与问答设计

- 向量模型：`shibing624/text2vec-base-chinese`
- 向量数据库：Chroma
- 生成模型：`deepseek-chat`
- 检索策略：BM25 + MMR 向量检索 + `EnsembleRetriever`
- 问答链类型：`RetrievalQA` + `stuff`

## 依赖说明

- `langchain`：问答链、提示词、检索流程编排
- `langchain-community`：PDF 加载器、BM25 检索器
- `langchain-openai`：调用 DeepSeek 兼容接口
- `langchain-huggingface`：加载中文嵌入模型
- `langchain-chroma`：Chroma 向量数据库封装
- `sentence-transformers`：向量模型底层依赖
- `pypdf`：PDF 解析
- `python-dotenv`：读取 `.env` 配置

## 注意事项

- `vector_db/` 属于本地生成文件，通常不建议提交
- 首次运行时会下载部分模型依赖，并在没有现成向量库时执行嵌入建库，耗时会比后续运行更长
- 如果更新了 `rag_docs/` 中的文档内容，建议删除旧的 `vector_db/` 后重新运行，以便重建索引
- 若安装新依赖后仍报导入错误，请确认当前使用的 Python 解释器就是安装依赖时对应的那个环境
- `rag_docs/` 中如果包含受版权、隐私或课程限定资料，请按实际情况决定是否上传

## 后续可扩展方向

- 增加更多文档格式支持，如 Markdown、TXT
- 支持自动检测文档变更并增量更新向量库
- 将命令行交互改为 Web 界面
- 增加可配置的检索策略与参数
- 补充异常处理、日志和测试
