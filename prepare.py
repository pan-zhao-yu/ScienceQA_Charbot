from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from tqdm import tqdm
import os
import shutil

os.environ["TOKENIZERS_PARALLELISM"] = "false" #停止一些不必要的警告
# 加载数据（字典结构，train split 是 dict 的 values）
dataset = load_dataset("derek-thomas/ScienceQA")
train_data = dataset["train"]
print(f"📦 Loaded {len(train_data)} training examples")

# 构建 LangChain Documents
documents = []
for example in tqdm(train_data):
    question = example.get("question", "").replace("\n", " ").strip().lower()
    choices = [c.replace("\n", " ").strip().lower() for c in example.get("choices", [])]
    answer = example.get("answer", "")
    lecture = example.get("lecture", "")
    hint = example.get("hint", "")
    solution = example.get("solution", "")
    subject = example.get("subject", "")
    topic = example.get("topic", "")
    skill = example.get("skill", "")
    category = example.get("category", "")
    task_type = example.get("task", "")

    # 仅用于embedding的内容，只包含question和choices
    embedding_content = f"[Question] {question}[Choices] {', '.join(choices)}"

    # 完整信息存入metadata中
    full_content = f"""
[Question] {question}
[Choices] {', '.join(choices)}
[Answer] {answer}
[Lecture] {lecture}
[Hint] {hint}
[Solution] {solution}
[Category] {category} | Subject: {subject} | Topic: {topic} | Skill: {skill}
    """.strip()

    metadata = {
        "full_content": full_content,
        "subject": subject,
        "topic": topic,
        "skill": skill,
        "task": task_type,
        "category": category,
    }

    documents.append(Document(page_content=embedding_content, metadata=metadata))

#Chunking 分块文本（按需修改参数）
splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0) #one doc per chunk
split_docs = splitter.split_documents(documents)
print(f"✂️ Split into {len(split_docs)} chunks")

# 嵌入模型（本地）
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# 输出前5个文档的 embedding content 和对应的 embedding 向量
print("embedding 文档及其向量：")
for doc in split_docs[:20]:
    # 这里调用 embed_query 生成向量（注意输入必须与构造时一致）
    vector = embedding_model.embed_query(doc.page_content)
    print("Embedding Content:")
    print(doc.page_content)
    # print("Embedding Vector:")
    # print(vector)
    print("-" * 50)


# 清理已有路径
save_path = "vectorstore/faiss_index"
if os.path.isfile(save_path):
    os.remove(save_path)
if os.path.isdir(save_path):
    shutil.rmtree(save_path)


# 构建 FAISS 向量库
vectorstore = FAISS.from_documents(split_docs, embedding_model)
print("FAISS vectorstore created")

# 保存向量库
os.makedirs("vectorstore", exist_ok=True)
vectorstore.save_local(save_path)
print(f"Vector store saved at {save_path}")