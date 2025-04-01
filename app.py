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

#local FAISS embedding
vectorstore = FAISS.load_local("vectorstore/faiss_index", embeddings=embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

#API KEY
llm = ChatOpenAI(model="gpt-4", temperature=0.2, api_key="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")

#Prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=get_prompt_template()
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
        print("\n📚 Retrieved Contexts:")
        for i, doc in enumerate(retrieved_docs):
            print(f"\n--- Chunk {i+1} ---")
            print(doc.page_content[:500])

        #RAG Chain Answer
        answer = rag_chain.invoke(query)
        print(f"\n🧠 Answer:\n{answer}")#use {answer['result']} to print the answer only
    except Exception as e:
        print(f"❌ Error: {e}")

        # compaire with pure LLM
        print("\n🤖 Compare Pure LLM Answer (No retrieval):")
        pure_answer = llm.invoke(query)
        print(pure_answer.content)

#sample question:
# 1. Which of these states is farthest north? "West Virginia","Louisiana","Arizona","Oklahoma"
# 2. Identify the question that Tom and Justin's experiment can best answer.","Do ping pong balls stop rolling along the ground sooner after being launched from a 30\u00b0 angle or a 45\u00b0 angle?","Do ping pong balls travel farther when launched from a 30\u00b0 angle compared to a 45\u00b0 angle?"
# 8. Is this a sentence fragment?\nDuring the construction of Mount Rushmore, approximately eight hundred million pounds of rock from the mountain to create the monument.","no","yes"

