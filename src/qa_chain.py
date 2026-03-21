import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
load_dotenv()

def get_qa_chain(vector_db, chunks):
    """
    Create a RetrievalQA chain using the provided vector database.
    """
    llm = ChatOpenAI(
        model='deepseek-chat', 
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
        openai_api_base='https://api.deepseek.com/v1', 
        temperature=0.2
    )

    template = """你是一个严谨的西安美食问答助手。请严格依据提供的上下文（Context）回答问题。

回答规则：
1. 只使用上下文中能明确支持的信息，不要补充常识性猜测。
2. 如果上下文信息不足或问题前提不成立，要直接说明。
3. 如果问题要求分类、归纳或列举，请尽量按条目化方式回答。
4. 优先保留文中的原始叫法和关键细节，不要随意改写成模糊概括。
5. 不要把不属于同一类别的内容强行合并。

上下文内容：
{context}

用户问题：
{question}

你的回答："""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5

    vector_retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 24}
    )

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=hybrid_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    return qa_chain
