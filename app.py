from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from chains.prompt_templates import get_prompt_template

# 加载向量库
embedding = OpenAIEmbeddings()
vectorstore = FAISS.load_local("vectorstore/faiss_index", embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 加载 LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)

# 自定义 Prompt
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=get_prompt_template()
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# 用户提问示例
while True:
    query = input("\nAsk a science question (or 'exit'): ")
    if query.lower() == "exit":
        break
    answer = rag_chain.run(query)
    print(f"\n🧠 Answer:\n{answer}")
