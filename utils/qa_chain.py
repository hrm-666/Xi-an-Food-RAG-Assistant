import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
load_dotenv()

def get_qa_chain(vector_db):
    """
    Create a RetrievalQA chain using the provided vector database.
    """
    llm = ChatOpenAI(
        model='deepseek-chat', 
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
        openai_api_base='https://api.deepseek.com/v1', 
        temperature=0.2 # 设置低一点，让回答更严谨
    )

    template = """你是一个专业的西安美食向导。请使用以下提供的上下文（Context）来回答用户的问题。
如果你在上下文中找不到答案，就直说不知道，不要尝试胡编乱造。
回答要求简明扼要，富有亲和力。

上下文内容：
{context}

用户问题：
{question}

你的回答："""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type = "stuff",
        retriever=vector_db.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 3,'fetch_k': 10}
            ), # 从向量数据库中检索最相关的5条信息
        return_source_documents=True, # 返回检索到的原始文档，方便后续分析
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    return qa_chain