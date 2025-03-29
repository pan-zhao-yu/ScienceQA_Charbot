from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from chains.prompt_templates import get_prompt_template
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI


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
        answer = rag_chain.run(query)
        print(f"\n🧠 Answer:\n{answer}")
    except Exception as e:
        print(f"❌ Error: {e}")

