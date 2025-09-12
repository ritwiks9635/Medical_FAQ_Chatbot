import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

CHROMA_DIR = "chroma_db"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("⚠️ GOOGLE_API_KEY not set. Please export it before running.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

def build_documents(df: pd.DataFrame):
    docs = []
    for _, row in df.iterrows():
        q = str(row["Question"]).strip()
        a = str(row["Answer"]).strip()
        qtype = str(row["qtype"]).strip() if "qtype" in df.columns else ""
        text = f"Question: {q}\nAnswer: {a}"
        metadata = {"question": q, "qtype": qtype}
        docs.append(Document(page_content=text, metadata=metadata))
    return docs

def build_vectorstore(csv_path="data/train.csv"):
    data = pd.read_csv(csv_path)
    raw_docs = build_documents(data)
    print(f"Built {len(raw_docs)} base documents")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    docs = splitter.split_documents(raw_docs)
    print(f"Split into {len(docs)} chunks")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=CHROMA_DIR
    )
    vectorstore.persist()
    print(f"Chroma DB built at: {CHROMA_DIR}, total docs stored: {len(docs)}")
    return vectorstore


def load_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding
    )


if __name__ == "__main__":
    build_vectorstore()