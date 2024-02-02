from fastapi import FastAPI, HTTPException

from rag import load_rag_pipeline,answer_question

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://10.0.2.2:8080"],  # Update with the correct address
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Paste the existing code here

# Load the RAG pipeline
qa_pipeline = load_rag_pipeline()

@app.post("/answer")
async def get_answer(text: str):
    print("Call from flutter")
    try:
        # Call the answer_question function with the provided text
        result = answer_question(text, qa_pipeline)
        print(result)
        return {"answer": result['result']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
