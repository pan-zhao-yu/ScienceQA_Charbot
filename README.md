# ScienceQA_Charbot

🧠 ScienceQA RAG Chatbot
This project is an interactive, retrieval-augmented generation (RAG) chatbot designed to answer science-related questions using real context from the ScienceQA dataset. It combines a FAISS vector database, sentence-transformer embeddings, and OpenAI's GPT-4.1 model through LangChain.
<img width="536" alt="Screenshot 2025-04-07 at 11 48 55" src="https://github.com/user-attachments/assets/fb49ad9f-e5a6-4c47-a4cf-a86c847c6c22" />



## Project Structure

```
.
├── vectorstore/               # Saved FAISS index
├── chains/
│   └── prompt_templates.py   # Prompt templates for different reasoning strategies
├── prepare.py      # Script to create FAISS index from ScienceQA dataset
├── app.py         # Main script to run the QA chatbot
└── README.md                 # You're here :)
```



to run the Chatbot:
1. Run the following script to install the requirements
   ```pip install -r requirements.txt```

2. Set OpenAI API key in the code to your own key(app.py, line 23)
   
3. Run the following script to load and process the ScienceQA dataset and build the FAISS vector index:
   ```python prepare.py```
   
4. Run the QA Chatbot:
   ```python app.py```

5. You can now type science questions like: which of these states is farthest north? [Choices] west virginia, louisiana, arizona, oklahoma

6. It will:

Retrieve relevant context from the FAISS database

Construct a thoughtful prompt using Chain-of-Thought reasoning

Use GPT-4.1 to provide a clear, student-friendly answer


