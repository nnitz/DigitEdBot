# FULL CODE

import chromadb
from chromadb import PersistentClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

# Step 1: Load text from file
with open("transcript_clean.txt", "r", encoding="utf-8") as file:
    text_to_chunk = file.read()

# Step 2: Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n"],
    chunk_size=200,
    chunk_overlap=50,
)

# Step 3: Split the document into chunks
chunks = text_splitter.create_documents([text_to_chunk])

# Step 4: Create Chroma client and collection
chroma_client = chromadb.PersistentClient(path="/Users/natalienitz/Desktop/DigitEd/RagBot/chromacollection")
collection = chroma_client.get_or_create_collection(
    name="test_bizint_chunks",
    metadata={"hnsw:space": "cosine"}
)

# Step 5: Add chunks to the collection
for idx, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk.page_content],
        ids=[f"chunk_{idx}"],
        metadatas=[{
            "chunk_index": idx,
            "source": "transcript_clean.txt"
        }]
    )

client = OpenAI(api_key="XXXXXXXXXXXXXXX")

# Step 6: Query the collection
query_text = "Chi era Devens?"
results = collection.query(query_texts=[query_text], n_results=3)

# Step 7: Print the results
print(f"\nTop results for query: '{query_text}'\n")
for i, doc in enumerate(results['documents'][0]):
    print(f"Result {i+1}:\n{doc}\n")