import argparse
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question using ONLY the information provided in the context.
Do not use outside knowledge.
DO NOT add any commentary.
If the answer cannot be found in the context, say:
"The answer is not explicitly stated in the provided context."

Context:
{context}

Question:
{question}

Answer:
"""

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("query_text",type=str,help='The query text.')
    args=parser.parse_args()
    query_text=args.query_text

    embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5",model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True})
    db = Chroma(embedding_function=embedding_function, persist_directory=CHROMA_PATH
    )

    results=db.similarity_search_with_score(query_text,k=8)
    filtered_results = [(doc, score) for doc, score in results if score < 0.8]
    if len(filtered_results) == 0:
        print("No sufficiently relevant results found.")
        return
    

    top_docs = filtered_results[:2]
    context_text = "\n\n---\n\n".join(doc.page_content for doc,_ in top_docs)
    prompt_template=ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt=prompt_template.format(context=context_text,question=query_text)
    print(prompt)

    model=ChatOllama(model="mistral")
    response=model.invoke(prompt)
    response_text=response.content

    sources=[doc.metadata.get("source",None) for doc,_score in results]
    formatted_response=f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()