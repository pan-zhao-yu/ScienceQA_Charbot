import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# 载入数据
data_path = Path("../data/scienceqa_train.json")
data = json.loads(data_path.read_text())

# 构造 LangChain 文档对象
docs = []
for example_id, example in data.items():
    content = f"""
Question: {example['question']}
Choices: {', '.join(example['choices'])}
Lecture: {example['lecture']}
Hint: {example['hint']}
Solution: {example['solution']}
Subject: {example['subject']}
Grade: {example['grade']}
"""
    metadata = {
        "id": example_id,
        "topic": example["topic"],
        "skill": example["skill"]
    }
    docs.append(Document(page_content=content.strip(), metadata=metadata))

# 文本分块（可调大小）
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# 嵌入 + 存入 FAISS
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embedding)
vectorstore.save_local("../vectorstore/faiss_index")
print("✅ FAISS index saved.")
