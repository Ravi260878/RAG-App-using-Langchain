
##Loading 2 pdf files from a folder into the project and reading the page no. and content
from langchain_community.document_loaders import PyPDFLoader

loader1 = PyPDFLoader(r"C:\RAG_DB_W2\C_Language.pdf")
loader2 = PyPDFLoader(r"C:\RAG_DB_W2\Computer fundamentals.pdf")

pages1 = loader1.load()
pages2 = loader2.load()

all_pages = pages1 + pages2

print("Total_pages:", len(all_pages))
#print(all_pages[0].page_content)

##Splitting the pdf into chunks using RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, # each chunk is about 1000 characters
    chunk_overlap = 150 # 150 characters repeat between chunks
)

chunks = splitter.split_documents(all_pages)

print("Totalchunks:", len(chunks))
#print(chunks[0].page_content)
#print(chunks[0].metadata)

##calling the embedding model and activating OPENAI_API_KEY
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

print("Openai embedding model is ready")

## Using FAISS (Fast vector store) embed the chunks, build FAISS indexing and then store it inside vector_store

vector_store = FAISS.from_documents(chunks,embeddings)

print("Embeded, indexed and stored in Vector_store")

##Storing the vector_store in local
vector_store.save_local(r"C:\RAG_DB_W2\My_FAISS")

print("Vector_file stored inside the local folder")

##Storing FAISS store from local to st.session_state to keep the index unique and to refer
##the indexed file by streamlit app while multiple reruns

import streamlit as st

if "vector_store" not in st.session_state:
    st.session_state.vector_store = FAISS.load_local(r"C:\RAG_DB_W2\My_FAISS",embeddings,allow_dangerous_deserialization=True)

print("FAISS store data moved to session state in streamlit")

    #Take the question from user in streamlit ui
    #Search in pdf chunk for get 3 relevant chunks
    # sharing the question and the 3 relevant chunks to chatmodel (OpenAI) for further processing and providing the final answer

from langchain_openai import ChatOpenAI

st.title("Ask your pdf's")
question = st.text_input("Enter your question here")

if question: #if the question is nonblank then further execution will happen
    with st.spinner("Searching the pdf and preparing the answer...."):
        docs = st.session_state.vector_store.similarity_search(question, k=3)

        context = "\n\n".join(doc.page_content for doc in docs) ##combining all 3 chunks into one "Context"

        llm = ChatOpenAI(model="gpt-4o-mini")

        ##Giving prompt to the llm to give answer only if the prompt given is satisfied

        response = llm.invoke(
            f"""Answer the question using only the context below.
If the context does not contain the answer, say you could not find it in the pdfs.

Context:
{context}

Question: {question}"""
        )

    st.write("Answer:")
    st.write(response.content)
    st.write("Sources:")

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown PDF") # prints the pdf name, if not available, then prints as unknown pdf
        page = doc.metadata.get("page") # gets the page no
        page_label = page +1 if isinstance(page, int) else "Unknown" # the page no. is integer it will add 1 as the pdf stored will start from 0, else it will print unknown
        st.write(f"Source {i}: {source}, page {page_label}")


