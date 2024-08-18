import gradio as gr
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

import json

with open('config.json', 'r') as file:
    config = json.load(file)

client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = config['nvidia_api_key']
    )

url = config['qdrant_url']
api_key = config['qdrant_api_key']

# Initialize Qdrant client
qdrant_client = QdrantClient(url=url, api_key=api_key)

def get_sentenceTF_embeddings(sentences):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings =[]
    for chunk in sentences:
        embeddings.append(model.encode(chunk))
    return embeddings

def Embed_stenteceTF(sentence):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(sentence)

messages = []
def custom_prompt(query: str):
    query_embedding_response = get_sentenceTF_embeddings(query)
    query_embedding = query_embedding_response[0].tolist()

    # Perform similarity search
    results = qdrant_client.search(
        collection_name="pneumonoBot",
        query_vector=query_embedding,

    )
    
    # Extract the page content from the results
    source_knowledge = "\n".join([x.payload['text'] for x in results])
    
    # Create the augmented prompt
    augment_prompt = f"""Answer the query,and dont mention the context explicitly:

    Additional Knowledge:
    {source_knowledge}

    Query: {query}"""
    
    return augment_prompt

# Load the model and feature extractor from the local directory
save_directory = "./vit_classification_pneumonobot"
model = ViTForImageClassification.from_pretrained(save_directory)
feature_extractor = ViTFeatureExtractor.from_pretrained(save_directory)

# Define the label mapping
labels = {0: 'Normal', 1: 'Pneumonia'}

def classify_image(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class_idx = outputs.logits.argmax(-1).item()
    predicted_label = labels[predicted_class_idx]
    return predicted_label

def model_inference( user_prompt, chat_history):
    if user_prompt["files"]:
        file_info = user_prompt["files"][0]
        file_path = file_info["path"]
        image = Image.open(file_path).convert("RGB")
        prediction = classify_image(image)
        return prediction
    else:
        query = user_prompt["text"]
        prompt = {"role":"system", "content": custom_prompt(query)}
        messages.append(prompt)

        res = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=messages,
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        stream=True
        )

        full_response = ""
        for chunk in res:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
        return full_response
        
    
# Create a chatbot interface
chatbot = gr.Chatbot(
    label="PneumonoBot",
    avatar_images=[None, None],
    show_copy_button=True,
    likeable=True,
    layout="panel",
    height=400,
)
output = gr.Textbox(label="Prompt")