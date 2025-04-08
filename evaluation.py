import re
import json
import logging
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 关闭不必要的警告
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


# -----------------------------
# 1. 定义辅助函数
# -----------------------------
def normalize_text(text):
    """
    归一化文本：移除换行符、多余空格，转换为小写
    """
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def extract_candidate_answer(answer_text, choices):
    """
    从模型输出中提取候选答案：
    如果输出中包含某个选项（忽略大小写），则返回该选项；
    否则直接返回经过 strip 的输出（转小写）。
    """
    ans_lower = answer_text.lower()
    for choice in choices:
        if choice.lower() in ans_lower:
            return choice.lower()
    return answer_text.strip().lower()


# -----------------------------
# 2. 定义评估 prompting 模板函数
# -----------------------------
def get_prompt_template_evaluation():
    return (
        "Context: {context}\n"
        "You are an evaluation assistant. Your task is to assess the answer provided for a multiple-choice question "
        "by comparing the candidate answer with the reference answer. Please follow these steps:\n\n"
        "1. Read the provided question, answer options, candidate answer, and reference answer.\n"
        "2. Compare the candidate answer with the reference answer. Ignore differences in case and punctuation.\n"
        "3. Determine if the candidate answer is correct based on its semantic and content similarity to the reference answer.\n"
        "4. Provide a brief explanation outlining the reasoning behind your evaluation.\n"
        "5. Finally, output the result in the exact JSON format shown below. Do not include any extra text or commentary.\n\n"
        "Expected JSON format:\n"
        "{\n"
        '  "question": "<question text>",\n'
        '  "predicted_answer": "<candidate answer>",\n'
        '  "reference_answer": "<reference answer>",\n'
        '  "is_correct": <true or false>,\n'
        '  "explanation": "<brief explanation>"\n'
        "}\n\n"
        "Now, evaluate the following information:\n"
        "Question: {question}\n"
        "Answer Options: {choices}\n"
        "Candidate Answer: {candidate_answer}\n"
        "Reference Answer: {reference_answer}\n\n"
        "Please return the evaluation result strictly in the JSON format specified above."
    )


# -----------------------------
# 3. 构建评估链（LLMChain）和检索链（RetrievalQA）
# -----------------------------
# 初始化嵌入模型（与 prepare 阶段保持一致）
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 加载本地 FAISS 向量库（请确保 prepare 脚本已生成并保存）
vectorstore = FAISS.load_local("vectorstore/faiss_index", embeddings=embedding_model,
                               allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 构造用于检索的 RAG 链，此处 prompt 要与 prepare 时生成 embedding 时的格式一致
retrieval_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Context: {context}\n[Question] {question}"
)
rag_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4", temperature=0.2, api_key="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"),
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": retrieval_prompt}
)

# 构造评估链，输入变量包括 context、question、choices、candidate_answer、reference_answer
evaluation_prompt = get_prompt_template_evaluation()
evaluation_chain = LLMChain(
    llm=ChatOpenAI(model="gpt-4", temperature=0.2, api_key="YOUR_API_KEY_HERE"),
    prompt=PromptTemplate(
        input_variables=["context", "question", "choices", "candidate_answer", "reference_answer"],
        template=evaluation_prompt
    )
)

# -----------------------------
# 4. 定义测试问题（示例）
#    如果你有一个 questions.json，也可以加载文件，这里手动定义几个样例问题
# -----------------------------
test_questions = [
    {
        "question": "Is this a sentence fragment? During the construction of Mount Rushmore, approximately eight hundred million pounds of rock from the mountain to create the monument.",
        "choices": ["no", "yes"],
        "reference_answer": "yes"
    },
    {
        "question": "Which of these states is farthest north? West Virginia, Louisiana, Arizona, Oklahoma.",
        "choices": ["west virginia", "louisiana", "arizona", "oklahoma"],
        "reference_answer": "west virginia"
    }
    # 可继续添加其他测试问题...
]

# -----------------------------
# 5. 对每个测试问题进行评估
# -----------------------------
correct_count = 0
total = len(test_questions)

for entry in test_questions:
    raw_question = entry["question"]
    choices = entry["choices"]
    reference_answer = normalize_text(entry["reference_answer"])

    # 构造查询时，保证格式与 prepare 时一致： "[Question] <question>\n[Choices] <choice1, choice2,...>"
    norm_question = normalize_text(raw_question)
    norm_choices = [normalize_text(choice) for choice in choices]
    query = f"[Question] {norm_question}\n[Choices] {', '.join(norm_choices)}"

    # 检索上下文（可帮助 LLM 评估时理解当前任务）
    retrieved_docs = retriever.get_relevant_documents(query)
    context_text = "\n".join([doc.metadata.get("full_content", doc.page_content) for doc in retrieved_docs])

    # 调用 RAG 检索链获取模型候选答案
    candidate_result = rag_chain.invoke(query)
    # 这里假设返回结果以字典格式存在 "result" 字段，否则直接使用返回的字符串
    candidate_answer_text = candidate_result.get("result", candidate_result).strip()
    candidate_answer = extract_candidate_answer(candidate_answer_text, norm_choices)

    # 使用评估链对比候选答案与参考答案，传入检索得到的上下文
    eval_input = {
        "context": context_text,
        "question": raw_question,
        "choices": ", ".join(choices),
        "candidate_answer": candidate_answer,
        "reference_answer": reference_answer
    }
    eval_output = evaluation_chain.run(eval_input)

    print("Evaluation result:")
    print(eval_output)
    print("-" * 60)

    # 这里为了统计准确率，采用简单的字符串比较（注意对大小写已归一化）
    if candidate_answer == reference_answer:
        correct_count += 1

accuracy = correct_count / total if total > 0 else 0
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
