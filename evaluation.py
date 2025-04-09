import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from chains.prompt_templates import get_prompt_template, get_prompt_template_COT
import logging

# -------------------------------
# 1. 数据准备：从数据集中抽取20个问题
# -------------------------------
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# 加载 ScienceQA 数据集
dataset = load_dataset("derek-thomas/ScienceQA")
train_data = dataset["train"]

dataset_questions = []
# 从数据集中抽取前20个样本的问题
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
# 2. 定义10个额外自选英文问题（可以根据需要选择开放题或者多选题）
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
# 3. 合并问题，并保存到 evaluation_questions.json 文件（便于后续查看）
# -------------------------------
all_questions = dataset_questions + custom_questions

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

# 定义问答提示模板（这里使用之前的模板）
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=get_prompt_template_COT()
)

# 构建 Retrieval-Augmented Generation (RAG) 问答链
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# -------------------------------
# 5. 循环提问，收集模型回答和检索的上下文信息
# -------------------------------
evaluation_results = []
for q in all_questions:
    question_text = q["question"]
    try:
        # 获取检索的文档上下文，列表中每项为对应文档的 page_content（可以根据需要进一步截断或加工）
        retrieved_docs = retriever.invoke(question_text)
        # 将每个文档内容简单提取出来，存入列表
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]

        # 使用 RAG 问答链得到模型回答
        answer = rag_chain.invoke(question_text)
        # 判断回答格式
        if isinstance(answer, dict) and "result" in answer:
            answer_text = answer["result"]
        elif hasattr(answer, "content"):
            answer_text = answer.content
        else:
            answer_text = str(answer)
    except Exception as e:
        answer_text = f"Error: {e}"
        retrieved_contexts = []

    result = {
        "id": q["id"],
        "source": q["source"],
        "question": question_text,
        "choices": q["choices"],
        "model_answer": answer_text,
        "retrieved_contexts": retrieved_contexts
    }
    evaluation_results.append(result)
    print(f"Processed question {q['id']}")

# -------------------------------
# 6. 保存问题、模型答案以及检索的上下文到 evaluation_4omini_COTPrompt_results.json 文件
# -------------------------------
with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
print("Evaluation results with retrieved contexts saved to evaluation_results.json")
