from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from chains.prompt_templates import get_prompt_template, get_prompt_template_COTAndFewShots, get_prompt_template_COT
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
import logging
#停止一些不必要的警告
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)



embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#配置api key使用命令：export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"
#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

#local FAISS embedding
vectorstore = FAISS.load_local("vectorstore/faiss_index", embeddings=embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

#API KEY
llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key="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")

#Prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=get_prompt_template_COT()
)

# construct RAG QA chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

#usesr input loop
while True:
    query = input("\nAsk a science question (or 'exit'): ")
    if query.lower() == "exit":
        break
    try:
        #print information retrieved from vectorstore
        retrieved_docs = retriever.get_relevant_documents(query)
        print("\n📚📚📚 Retrieved Contexts:")
        for i, doc in enumerate(retrieved_docs):
            print(f"\n--- Chunk {i+1} ---")
            print(doc.metadata.get("full_content", doc.page_content))

        # 将检索到的内容拼接成上下文文本
        context_text = "\n".join(
            [doc.metadata.get("full_content", doc.page_content) for doc in retrieved_docs]
        )

        # 使用 prompt 模板生成最终的 prompt 内容，并打印出来
        prompt_content = prompt.format(context=context_text, question=query)
        print("\n📝📝📝 Final Prompt:")
        print(prompt_content)

        #RAG Chain Answer
        answer = rag_chain.invoke(query)
        #use print(f"\n🧠 Answer:\n{answer}") to print full output
        #use print(f"\n🧠 Answer:\n{answer['result']}") to print the answer only
        print(f"\n🧠🧠🧠 Answer:\n{answer['result']}")
    except Exception as e:
        print(f"❌ Error: {e}")

