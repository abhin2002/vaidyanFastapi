import os
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS



import time

import os
from dotenv import load_dotenv

load_dotenv()

# Supress warnings
import warnings
warnings.filterwarnings("ignore")

def load_rag_pipeline(name = "faiss_index"):
    # Load the index
    db = FAISS.load_local(name, OpenAIEmbeddings())
    # Expose this index in a retriever interface
    retriever = db.as_retriever()
    # Create a chain to answer questions
    return RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
    )


def answer_question(question,qa):
    context = "You are a helpful doctor that would provide patients with first aid information. Give them detailed and simple instructionsso that. The input by the patient given seperated by hyphens ---{0}---"
    answer = qa(context.format(question))
    return answer

if __name__ == "__main__":
    print("Building RAG pipeline...")
    start = time.time()
    # Load the RAG pipeline
    qa = load_rag_pipeline()
    print("Building took {0}s".format(time.time()-start))
    print("Done building RAG pipeline")
    print("Answering question...")
    start = time.time()
    # Answer a question
    ans = answer_question("What is the first aid for a heart attack?",qa)
    print(ans)
    print(ans['result'])
    print("Answering took {0}s".format(time.time()-start))
    print("Done answering question")

        

