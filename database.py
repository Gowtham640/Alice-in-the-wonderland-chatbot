from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader




data_path="data/books"

CHROMA_PATH = "chroma"

def main():
    generate_data_store()

def generate_data_store():
    documents=load_documents()
    chunks=split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(
        data_path,
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    return loader.load()

def split_text(documents: list[Document]):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=120,
        length_function=len,
        add_start_index=True,
    )
    chunks=text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document=chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

def save_to_chroma(chunks:list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5",model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True})
    db = Chroma.from_documents(
        chunks, embedding, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()