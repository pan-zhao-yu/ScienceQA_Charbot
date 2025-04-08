from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from tqdm import tqdm
import os
import shutil

# 停止部分不必要的警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 加载 ScienceQA 数据集
dataset = load_dataset("derek-thomas/ScienceQA")
train_data = dataset["train"]
print(f"📦 Loaded {len(train_data)} training examples")

# 构建 LangChain 文档
documents = []
for example in tqdm(train_data):
    question = example.get("question", "")
    choices = example.get("choices", [])
    lecture = example.get("lecture", "")
    hint = example.get("hint", "")
    solution = example.get("solution", "")
    subject = example.get("subject", "")
    topic = example.get("topic", "")
    skill = example.get("skill", "")
    category = example.get("category", "")
    task_type = example.get("task", "")
    answer_index = example.get("answer", None)
    image = example.get("image", "")

    # 构造更加结构化、详细的文档内容，这样能保证嵌入时覆盖更全面的信息
    content = f"""
【Subject】: {subject}    【Topic】: {topic}    【Skill】: {skill}    【Category】: {category}
【Task】: {task_type}    【Answer Index】: {answer_index}
-------------------------------
[Question]
{question}

[Choices]
{', '.join(choices)}

-------------------------------
[Lecture]
{lecture}

-------------------------------
[Hint]
{hint}

-------------------------------
[Solution]
{solution}
    """.strip()

    # 增加 metadata 信息以便后续检索或辅助筛选
    metadata = {
        "subject": subject,
        "topic": topic,
        "skill": skill,
        "task": task_type,
        "category": category,
        "answer": answer_index,
        "image": image,
    }

    documents.append(Document(page_content=content, metadata=metadata))

# 使用较小的 chunk_size 以及适当增加 chunk_overlap 来确保每个 chunk 内能保留完整的上下文信息
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(documents)
print(f"✂️ Split into {len(split_docs)} chunks")

# 初始化本地嵌入模型（使用 HuggingFace 的 sentence-transformers 模型）
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 清理已有的向量库文件夹
save_path = "vectorstore/faiss_index"
if os.path.isfile(save_path):
    os.remove(save_path)
if os.path.isdir(save_path):
    shutil.rmtree(save_path)

# 使用 FAISS 构建向量库
vectorstore = FAISS.from_documents(split_docs, embedding_model)
print("FAISS vectorstore created")

# 保存向量库
os.makedirs("vectorstore", exist_ok=True)
vectorstore.save_local(save_path)
print(f"Vector store saved at {save_path}")
