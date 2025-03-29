def get_prompt_template():
    return """You are a helpful science tutor for elementary and high school students.
Use the following lecture content and explanations to answer the question carefully.

Context:
{context}

Question:
{question}

Answer (explain step-by-step and give final answer):"""
