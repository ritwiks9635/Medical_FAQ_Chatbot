# 🩺 Medical FAQ Chatbot (RAG + Gemini)

## 📌 Project Overview
Hospitals and clinics often receive repetitive questions from patients regarding symptoms, treatments, and general medical knowledge.  
This project implements a **Retrieval-Augmented Generation (RAG) chatbot** that can answer medical FAQs accurately using a knowledge base.

The chatbot:
- Retrieves relevant FAQs from a medical dataset using **Gemini embeddings + ChromaDB**.
- Generates **natural, safe answers** with **Gemini LLM**.
- Provides an easy-to-use **Streamlit interface**.

---

## ⚙️ Tech Stack
- **LangChain** → RAG pipeline & integration.  
- **Gemini API** → embeddings + text generation.  
- **ChromaDB** → vector database for semantic search.  
- **Streamlit** → user-friendly chatbot interface.  
- **Pandas** → CSV dataset handling.  

---

## 📂 Project Structure
```

medical-rag-chatbot/
│── data/medical\_faqs.csv       # Sample dataset (100 FAQs)
│── vectorstore.py              # Build & load vector database
│── rag.py                      # RAG pipeline (retrieval + generation)
│── app.py                      # Streamlit app frontend
│── requirements.txt            # Dependencies
│── README.md                   # Project documentation

````

---

## 🚀 Setup Instructions

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd medical-rag-chatbot
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Gemini API Key

Export your API key as an environment variable:

```bash
export GOOGLE_API_KEY="your_api_key_here"   
set GOOGLE_API_KEY="your_api_key_here"     
```

---

## 🏗️ Build Vectorstore

Before running the chatbot, preprocess the dataset and create embeddings:

```bash
python -c "from vectorstore import build_vectorstore; build_vectorstore('data/medical_faqs.csv')"
```

This will create a `chroma_db/` folder with embeddings stored.

---

## 💬 Run the Chatbot

Start the Streamlit app:

```bash
streamlit run app.py
```

Open the provided **local URL** in your browser (default: `http://localhost:8501`).

---

## 📊 Example Queries

* *"What are the early symptoms of diabetes?"*
* *"Can children take paracetamol?"*
* *"What foods are good for heart health?"*


```
