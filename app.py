import streamlit as st
from rag import generate_answer

st.set_page_config(page_title="Medical RAG Chatbot", page_icon="🩺")

st.title("🩺 Medical FAQ Chatbot (RAG + Gemini)")
st.write("Ask your medical questions below (based on the knowledge base).")

user_input = st.text_input("Your Question:")

if user_input:
    with st.spinner("Thinking..."):
        try:
            answer = generate_answer(user_input)
            st.success(answer)
        except Exception as e:
            st.error(f"Error: {str(e)}")