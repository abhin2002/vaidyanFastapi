from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, CodeGenTokenizerFast as Tokenizer
from PIL import Image
from rag import load_rag_pipeline, answer_question

app = Flask(__name__)

model_id = "vikhyatk/moondream1"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
tokenizer = Tokenizer.from_pretrained(model_id)

# Load RAG pipeline for recommended first aid
qa = load_rag_pipeline()

def analyze_image():
    try:
        # Receive image file from the client
        image_file = request.files['image']
        
        # Open image and encode it
        image = Image.open(image_file)
        enc_image = model.encode_image(image)
        
        # Get prompt from client
        prompt = request.form.get('prompt', "You are an emergency assistant and you have to give a short description on this accident situation that the person is in")
        
        # Get response from model
        response = model.answer_question(enc_image, prompt, tokenizer)
        
        # Get recommended first aid using RAG pipeline
        ans = answer_question(response, qa)

        return jsonify({'response': response, 'recommended_first_aid': ans})

    except Exception as e:
        return jsonify({'error': str(e)})


