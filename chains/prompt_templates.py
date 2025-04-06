# def get_prompt_template():
#     return """You are a helpful science tutor for elementary and high school students.
# Use the following lecture content and explanations to answer the question carefully.
#
# Context:
# {context}
#
# Question:
# {question}
#
# Answer (explain step-by-step and give final answer):"""

# v2

def get_prompt_template():
    return """You are a helpful and knowledgeable science tutor for students from elementary to high school.
Using the provided context — including lecture notes, hints, possible answer choices, and background — answer the question step by step.

If there are multiple choices, select the most correct one and explain why.

--- Context ---
{context}
--- End of Context ---

Question:
{question}

Answer (explain clearly and give the final answer at the end):"""


# def get_prompt_template():