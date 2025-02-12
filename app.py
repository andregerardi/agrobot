import fitz  # PyMuPDF
import streamlit as st
from openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os


# Inicializar o cliente OpenAI/Databricks
client = OpenAI(
    api_key=API_KEY_ANDRE,
    base_url="https://fgv-pocs-genie.cloud.databricks.com/serving-endpoints"
)

# Inicializar ChromaDB
db = Chroma(collection_name="pdf_chunks")

# Função para gerar embeddings via Databricks
def get_embeddings(text):
    response = client.embeddings.create(
        model="databricks-meta-llama-3-1-405b-instruct",
        input=[text]
    )
    return response.data[0].embedding  # Retorna o vetor gerado

# Função para extrair texto de um arquivo PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Função para dividir o PDF e armazenar embeddings no ChromaDB
def store_pdf_in_chromadb(pdf_text):
    text_splitter = CharacterTextSplitter(chunk_size=300, separator="\n")
    chunks = text_splitter.split_text(pdf_text)

    for i, chunk in enumerate(chunks):
        embedding = get_embeddings(chunk)  # Gera embedding via Databricks
        db.add_texts([chunk], metadatas=[{"index": i}], embeddings=[embedding])

# Função para buscar trechos mais relevantes no ChromaDB
def retrieve_relevant_text(question):
    query_embedding = get_embeddings(question)  # Gera embedding da pergunta
    results = db.similarity_search_by_vector(query_embedding, k=3)  # Busca no ChromaDB
    return "\n\n".join([doc.page_content for doc in results])

# Função para responder perguntas com base no PDF
def ask_question(question, history=[]):
    relevant_text = retrieve_relevant_text(question)

    messages = [
        {
            "role": "system",
            "content": "Você é um assistente técnico agrícola. Suas respostas devem ser diretas, objetivas e fáceis de entender para agricultores."
        }
    ]
    
    messages.extend(history)

    messages.append({"role": "user", "content": f"Baseado no seguinte conteúdo do PDF:\n\n{relevant_text}\n\nPergunta: {question}"})

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="databricks-meta-llama-3-1-405b-instruct",
        max_tokens=750
    )
    
    response = chat_completion.choices[0].message.content

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": response})

    return response, history

# Interface do chatbot no Streamlit
def main():
    st.title("Agrônomo Virtual 🤖")

    uploaded_file = st.file_uploader("Escolha um arquivo PDF", type=["pdf"])

    if uploaded_file is not None:
        pdf_text = extract_text_from_pdf(uploaded_file)

        if not pdf_text.strip():
            st.error("O PDF não contém texto legível.")
            return

        store_pdf_in_chromadb(pdf_text)  # Armazena os embeddings do PDF

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
                response, history = ask_question(question, st.session_state.history)
                st.session_state.history = history

            with st.chat_message("assistant"):
                st.markdown(response)

if __name__ == "__main__":
    main()
