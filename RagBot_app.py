import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from openai import OpenAI

# Securely load API key from secrets
api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

st.set_page_config(page_title="Digit'Ed Chatbot", layout="centered")
st.title("Digit'Ed Chatbot")

st.markdown('<div class="logo">Digit’Ed</div>', unsafe_allow_html=True)
st.markdown("Ask your question about the transcript:")

user_query = st.text_input("Type your question")
n_results = st.slider("How many results to retrieve?", min_value=1, max_value=5, value=3)

@st.cache_data(show_spinner="Embedding and indexing transcript...")
def embed_and_store():
    with open("transcript_clean.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"], chunk_size=300, chunk_overlap=50
    )
    chunks = text_splitter.split_text(raw_text)
    documents = [Document(page_content=chunk) for chunk in chunks]

    embedding = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_documents(documents, embedding)
    return db

db = embed_and_store()

system_prompt = "You are a helpful RAG assistant using course materials to answer questions."

def make_rag_prompt(query, result_str):
    return f"""Instructions:
Your task is to answer the following user question. Use the Search Results provided below to construct an accurate response. If the answer is not in the results, say "I don't know."

User question:
{query}

Search Results:
{result_str}

Your answer:
"""


def get_completion(user_prompt, system_prompt, model="gpt-4"):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return completion.choices[0].message.content

if st.button("Get Answer") and user_query:
    result = db.similarity_search(user_query, k=n_results)
    result_str = "\n\n".join([doc.page_content for doc in result])

    full_prompt = make_rag_prompt(user_query, result_str)
    answer = get_completion(full_prompt, system_prompt)

    st.markdown("### Answer:")
    st.write(answer)