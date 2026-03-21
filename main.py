import os
from src.loader import load_tech_docs
from src.splitter import split_documents
from src.database import create_db
from src.qa_chain import get_qa_chain

def main():
    print("Starting the application...")
    docs = load_tech_docs("./rag_docs")
    chunks = split_documents(docs)
    vector_db = create_db(chunks)
    qa_chain = get_qa_chain(vector_db, chunks)

    print("\n" + "="*40)
    print("西安美食助手已就绪！")
    print("输入 'exit' 或 'quit' 退出程序。")
    print("="*40)

    while True:
        query = input("\n请输入西安美食相关的问题：")
        if query.lower() in ["exit", "quit"]:
            print("感谢使用西安美食助手！")
            break
        
        if not query:
            continue

        print("正在查询，请稍候...")

        try:
            response = qa_chain.invoke({"query": query})
            print(f"\n回答：{response['result']}")
            print("\n参考来源：")
            sources = set()
            for doc in response["source_documents"]:
                source_info = f"- {doc.metadata.get('source')} (第 {doc.metadata.get('page', '?')} 页)"
                sources.add(source_info)
            
            for s in sources:
                print(s)

        except Exception as e:
            print(f"查询过程中发生错误：{e}")

if __name__ == "__main__":
    main()
