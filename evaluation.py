import json
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from chains.prompt_templates import get_prompt_template
import logging

# -------------------------------
# 1. 数据准备：抽取 20 个数据集中的问题
# -------------------------------
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# 加载 ScienceQA 数据集
dataset = load_dataset("derek-thomas/ScienceQA")
train_data = dataset["train"]

dataset_questions = []
# 从数据集中抽取前20个问题
for idx, example in enumerate(train_data):
    if idx >= 20:
        break
    q = {
        "id": f"dataset_{idx + 1}",
        "question": example.get("question", ""),
        "choices": example.get("choices", []),
        "source": "dataset"
    }
    dataset_questions.append(q)

# -------------------------------
# 2. 定义 10 个额外的自选问题（英文）
# -------------------------------
custom_questions = [
    {
        "id": "custom_1",
        "question": "Which part of a map typically indicates the direction of north?",
        "choices": ["Compass rose", "Scale bar", "Legend", "Graticule"],
        "source": "custom"
    },
    {
        "id": "custom_2",
        "question": "How does a compass rose help in reading maps?",
        "choices": [],
        "source": "custom"
    },
    {
        "id": "custom_3",
        "question": "Why might experiments use a control group?",
        "choices": [],
        "source": "custom"
    },
    {
        "id": "custom_4",
        "question": "Which element is not typically found on a traditional map?",
        "choices": ["North arrow", "Legend", "Compass", "GPS coordinates"],
        "source": "custom"
    },
    {
        "id": "custom_5",
        "question": "Describe the importance of consistent orientation in map reading.",
        "choices": [],
        "source": "custom"
    },
    {
        "id": "custom_6",
        "question": "What is one advantage of using multiple data chunks for embedding in a vector database?",
        "choices": [],
        "source": "custom"
    },
    {
        "id": "custom_7",
        "question": "Identify the benefit of using a chain-of-thought approach when answering complex questions.",
        "choices": [],
        "source": "custom"
    },
    {
        "id": "custom_8",
        "question": "Which factor is least likely to affect plant growth in a controlled experiment?",
        "choices": ["Soil type", "Watering schedule", "Ambient noise", "Light exposure"],
        "source": "custom"
    },
    {
        "id": "custom_9",
        "question": "Explain how artificial intelligence can enhance scientific inquiry.",
        "choices": [],
        "source": "custom"
    },
    {
        "id": "custom_10",
        "question": "Discuss the role of context in retrieving relevant information from a knowledge base.",
        "choices": [],
        "source": "custom"
    }
]

# -------------------------------
# 3. 合并问题列表并生成 JSON 文件
# -------------------------------
all_questions = dataset_questions + custom_questions

# 保存问题到 evaluation_questions.json
with open("evaluation_questions.json", "w", encoding="utf-8") as f:
    json.dump(all_questions, f, ensure_ascii=False, indent=4)
print("Evaluation questions saved to evaluation_questions.json")

# -------------------------------
# 4. 初始化向量库及问答链
# -------------------------------
# 使用与 prepare.py 中相同的嵌入模型
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# 从本地加载之前构建好的 FAISS 向量库
vectorstore = FAISS.load_local("vectorstore/faiss_index", embeddings=embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 请确保替换下面 API Key 为有效值
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key="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")

# 定义问答提示模板（使用之前定义的模板）
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=get_prompt_template()
)

# 构建 Retrieval-Augmented Generation (RAG) 问答链
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# -------------------------------
# 5. 循环提问，并收集模型的回答
# -------------------------------
evaluation_results = []
for q in all_questions:
    question_text = q["question"]
    try:
        # 可选：获得检索的文档上下文（用于调试或后续分析）
        retrieved_docs = retriever.get_relevant_documents(question_text)
        # 使用 RAG Chain 得到答案
        answer = rag_chain.invoke(question_text)
        # 处理回答格式：可能是字典（包含 "result" 键）、对象或字符串
        if isinstance(answer, dict) and "result" in answer:
            answer_text = answer["result"]
        elif hasattr(answer, "content"):
            answer_text = answer.content
        else:
            answer_text = str(answer)
    except Exception as e:
        answer_text = f"Error: {e}"

    result = {
        "id": q["id"],
        "source": q["source"],
        "question": question_text,
        "choices": q["choices"],
        "model_answer": answer_text
    }
    evaluation_results.append(result)
    print(f"Processed question {q['id']}")

# -------------------------------
# 6. 保存提问及答案结果到 evaluation_4omini_defaultPrompt_results.json
# -------------------------------
with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
print("Evaluation results saved to evaluation_results.json")
