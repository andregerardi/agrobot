import fitz  # PyMuPDF
import streamlit as st
from openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY_ANDRE = st.secrets["auth_token"]

# Corrigida: aceita arquivo binário (ex: BytesIO)
def extract_text_from_pdf(file_obj):
    doc = fitz.open(stream=file_obj.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Função simples para ranquear chunks por presença de palavras da pergunta
def get_most_relevant_chunks(question, chunks, top_n=4):
    question_words = set(question.lower().split())
    chunk_scores = []

    for i, chunk in enumerate(chunks):
        chunk_words = set(chunk.lower().split())
        common_words = question_words.intersection(chunk_words)
        score = len(common_words)
        chunk_scores.append((i, score))

    # Ordenar pelos scores decrescentes
    sorted_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in sorted_chunks[:top_n]]
    return top_indices

# Função para enviar a pergunta e obter resposta considerando histórico
def ask_question_from_pdf(pdf_text, question, history=[]):
    try:
        text_splitter = CharacterTextSplitter(chunk_size=500, separator="\n")
        chunks = text_splitter.split_text(pdf_text)
    except Exception as e:
        st.error(f"Erro ao dividir o texto: {str(e)}")
        return "", history

    # Obter os chunks mais relevantes (simples)
    similar_indices = get_most_relevant_chunks(question, chunks)

    client = OpenAI(
        api_key=API_KEY_ANDRE,
        base_url="https://fgv-pocs-genie.cloud.databricks.com/serving-endpoints"
    )

    messages = [{
        "role": "system", 
        "content": (
            "Você é um assistente técnico agrícola. "
            "Suas respostas devem ser fáceis de entender e voltadas para agricultores. "
            "Responda apenas com base no pdf fornecido."
        )
    }]

    messages.extend(history)

    relevant_chunks = "\n\n".join([chunks[i] for i in similar_indices])
    messages.append({"role": "user", "content": f"Baseado no seguinte conteúdo do PDF: {relevant_chunks}\n\nPergunta: {question}"})

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="databricks-meta-llama-3-3-70b-instruct",
        max_tokens=1024
    )

    response = chat_completion.choices[0].message.content

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": response})

    return response, history

# Interface do chatbot no Streamlit
def main():
    st.title("Agrônomo Virtual 🤖")

    uploaded_file = st.file_uploader("Envie o PDF", type="pdf")

    if uploaded_file:
        pdf_text = extract_text_from_pdf(uploaded_file)
        #st.text_area("Texto extraído", pdf_text, height=400)

        if not pdf_text.strip():
            st.error("O PDF não contém texto legível.")
            return

        if 'history' not in st.session_state:
            st.session_state.history = []

        st.subheader("Chat")

        for message in st.session_state.history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Digite sua pergunta sobre o PDF...")

        if question:
            with st.chat_message("user"):
                st.markdown(question)

            with st.spinner("Processando..."):
                response, history = ask_question_from_pdf(pdf_text, question, st.session_state.history)
                st.session_state.history = history

            with st.chat_message("assistant"):
                st.markdown(response)

if __name__ == "__main__":
    main()
