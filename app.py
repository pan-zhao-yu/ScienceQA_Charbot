from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from chains.prompt_templates import get_prompt_template
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
import logging
#停止一些不必要的警告
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)



embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#配置api key使用命令：export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"
#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

# 加载本地向量库
vectorstore = FAISS.load_local("vectorstore/faiss_index", embeddings=embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 加载大模型（这里仍然是 OpenAI 的 GPT-4，你可以按需替换）
llm = ChatOpenAI(model="gpt-4", temperature=0.2, api_key="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")

# 自定义 Prompt 模板
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=get_prompt_template()
)

# 构建 RAG QA chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# 用户提问循环
while True:
    query = input("\nAsk a science question (or 'exit'): ")
    if query.lower() == "exit":
        break
    try:
        #print检索到的向量库内容
        retrieved_docs = retriever.get_relevant_documents(query)
        print("\n📚 Retrieved Contexts:")
        for i, doc in enumerate(retrieved_docs):
            print(f"\n--- Chunk {i+1} ---")
            print(doc.page_content[:500])

        #然后调用 RAG
        answer = rag_chain.invoke(query)
        print(f"\n🧠 Answer:\n{answer}")#使用{answer['result']}可以只展示答案
    except Exception as e:
        print(f"❌ Error: {e}")

        # 对比纯LLM回答
        print("\n🤖 Compare Pure LLM Answer (No retrieval):")
        pure_answer = llm.invoke(query)
        print(pure_answer.content)

#sample question:
# 1. Which of these states is farthest north? "West Virginia","Louisiana","Arizona","Oklahoma"
# 8. Is this a sentence fragment?\nDuring the construction of Mount Rushmore, approximately eight hundred million pounds of rock from the mountain to create the monument.","no","yes"

