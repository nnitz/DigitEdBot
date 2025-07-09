import streamlit as st
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

files = [
    {
        "title": "Principles of Data Science",
        "source_url": "https://openstax.org/books/principles-data-science/pages/1-introduction",
        "filename": "BusinessEthics-OP.txt"
  },
  {
        "title": "Introduction to Business Ethics",
        "source_url": "https://openstax.org/books/business-ethics/pages/1-introduction",
        "filename": "Principles-of-Data-Science-WEB.txt"
  }
]

# Instantiate a Chroma persistent client
client = chromadb.PersistentClient("./")

## YOUR SOLUTION HERE ##
collection = client.get_or_create_collection(name = "Learning_Assistant", 
                                             metadata = {"hnsw:space": "cosine"})


#Read first file content
with open(f"./{files[0]['filename']}", "r") as file:
  content = file.read()
# Create a text splitter
## YOUR SOLUTION HERE ##
text_splitter = RecursiveCharacterTextSplitter(separators = ["\n\n", "\n", ". ", "? ", "! "], chunk_size = 1500, chunk_overlap = 200)





# Split the 'content' into chunks
chunks = text_splitter.create_documents([content])

# Print the first document
chunks[:1]


#Create empty lists to store each document, metadata, and id
documents = []
metadatas = []
ids = []

#Loop through each file in files
for file_info in files:
    with open(f"./{file_info['filename']}", "r") as file:
        content = file.read()
        #Use text_splitter to create documents
        chunks = text_splitter.create_documents([content])
        #iterate over every chunk
        for index, chunk in enumerate(chunks):
            #Append to metadata list with "title", "source_url", and "index"
            metadatas.append({
                "title": file_info["title"],
                "source_url": file_info["source_url"],
                "chunk_idx": index
            })
            #Append to ids each index
            ids.append(f"{file_info['filename']}_{index}")
            
            #Append to documents each chunk.page_content
            documents.append(chunk.page_content)
            
#Add all documents to the collection
collection.add(documents=documents, metadatas=metadatas, ids=ids)

#Verify documents were added to collection with a sample query
query_text = "What can you tell me about the value of data for companies?"
results = collection.query(query_texts=[query_text], n_results=3)

st.write(f"Number of documents in collection: {collection.count()}")