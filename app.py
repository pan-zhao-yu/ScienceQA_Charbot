import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from chains.prompt_templates import get_prompt_template, get_prompt_template_COTAndFewShots, get_prompt_template_COT
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#to config api key, use command：export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"
#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

#local FAISS embedding
vectorstore = FAISS.load_local("vectorstore/faiss_index", embeddings=embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

#API KEY
llm = ChatOpenAI(model="gpt-4.1", temperature=0.2, api_key="API_KEY")

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
        retrieved_docs = retriever.invoke(query)
        print("\n📚📚📚 Retrieved Contexts:")
        for i, doc in enumerate(retrieved_docs):
            print(f"\n--- Chunk {i+1} ---")
            print(doc.metadata.get("full_content", doc.page_content))

        # retrieve context text
        context_text = "\n".join(
            [doc.metadata.get("full_content", doc.page_content) for doc in retrieved_docs]
        )

        # print final prompt input
        prompt_content = prompt.format(context=context_text, question=query)
        print("\n📝📝📝 Final Prompt:")
        print(prompt_content)

        #RAG Chain Answer
        answer = rag_chain.invoke(query)
        print(f"\n🧠🧠🧠 Answer:\n{answer['result']}")
    except Exception as e:
        print(f"❌ Error: {e}")

