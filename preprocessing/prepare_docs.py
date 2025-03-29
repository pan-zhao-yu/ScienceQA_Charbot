# preprocessing/prepare_docs.py

from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from tqdm import tqdm

# === Load ScienceQA from HuggingFace ===
dataset = load_dataset("scienceqa/scienceqa", split="train")

# === Convert to LangChain Documents ===
docs = []
for example in tqdm(dataset):
    question = example["question"]
    choices = example["choices"]
    lecture = example["lecture"] or ""
    explanation = example["explanation"] or ""
    hint = example["hint"] or ""

    content = f"""Question: {question}
Choices: {', '.join(choices)}
Lecture: {lecture}
Hint: {hint}
Explanation: {explanation}
"""

    metadata = {
        "grade": example["grade"],
        "subject": example["subject"],
        "topic": example["topic"],
        "skill": example["skill"]
    }

    docs.append(Document(page_content=content.strip(), metadata=metadata))

# === Text Split ===
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# === Embedding & Save to FAISS ===
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embedding)
vectorstore.save_local("../vectorstore/faiss_index")

print("✅ Vector store created and saved.")
