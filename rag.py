import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from vectorstore import load_vectorstore

vectorstore = load_vectorstore()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("⚠️ GOOGLE_API_KEY not set. Please export it before running.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


gemini_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.4,
        google_api_key=GOOGLE_API_KEY
    )

def build_prompt(question: str, docs: list[Document]) -> str:
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""
You are a helpful, careful medical assistant chatbot.

Use ONLY the following context to answer the user question. 
If the answer is not in the context, say: 
"I don't have enough information from the knowledge base. Please consult a healthcare professional."

Context:
{context}

User Question: {question}

Answer clearly and concisely.
"""
    return prompt

def generate_answer(question: str, docs: list[Document],
                    model, temperature: float = 0.0) -> str:
    prompt = build_prompt(question, docs)
    response = model.invoke(prompt) 
    return response.content 

def answer_query(question: str):
    docs = retriever.invoke(question)
    answer = generate_answer(question, docs, gemini_model)
    
    return answer