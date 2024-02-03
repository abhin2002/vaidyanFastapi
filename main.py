import os


from rag import load_rag_pipeline, answer_question
from flask import Flask, request, jsonify
# from image2text import analyze_image

app = Flask(__name__)

# Load the question-answering pipeline
qa_pipeline = load_rag_pipeline()

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data['question']

        # Answer the question using the question-answering pipeline
        answer = answer_question(question, qa_pipeline)

        response = {
            'question': question,
            'answer': answer["result"],
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# @app.route('/analyze_image', methods=['POST'])
# def get_text_from_image():
#     return analyze_image()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))