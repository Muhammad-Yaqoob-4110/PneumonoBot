import gradio as gr
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch
import json
import lamini
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from typing import List
from rank_bm25 import BM25Okapi
import tiktoken
from bpemb import BPEmb

with open('config.json', 'r') as file:
    config = json.load(file)

def loadModel():
    lamini.api_key = config['lamini_api_key']
    llm = lamini.Lamini("meta-llama/Meta-Llama-3-8B-Instruct")
    return llm

def read_faiss_indices(data_filename='faiss_data.pkl'):
    # Load the data from the pickle file
    with open(data_filename, 'rb') as file:
        data = pickle.load(file)

    # Load the FAISS indices from their serialized files
    index_ip = faiss.read_index(data['index_ip_filename'])
    index_hnsw = faiss.read_index(data['index_hnsw_filename'])

    # Update the data dictionary with the loaded indices
    data['index_ip'] = index_ip
    data['index_hnsw'] = index_hnsw

    return data

def get_sentenceTF_embeddings(sentences):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings =[]
    for chunk in sentences:
        embeddings.append(model.encode(chunk))
    # print(len(embeddings))
    return embeddings

def Embed_stenteceTF(sentence):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(sentence)

def search_top_k_sentences_cos(data, input,input_embedding,threshold = 0.0 ,k=6):
    index_ip = data['index_ip']   # Euclidean distance index (IndexFlatL2)
    sentences = data['sentences']
    #print(input)
    # Convert the input embedding to a numpy array
    input_embedding_array = np.array([input_embedding])

    # Perform the cosine similarity search
    distances_ip, indices_ip = index_ip.search(input_embedding_array, k)
    top_k_indices_ip = indices_ip[0][distances_ip[0] >= threshold]
    # Get the corresponding sentences for the top k embeddings
    top_k_distances_ip = distances_ip[0]

    # Get the corresponding sentences and scores for the top k embeddings
    top_k_sentences = [sentences[i] for i in top_k_indices_ip]

    return top_k_sentences

def search_top_k_sentences(data,input_embedding,input_text, k, preprocess_func,threshold = 0.0 ):
    # Get the Faiss indices and sentences from the data dictionary
    index_ip = data['index_ip']  # Cosine similarity index (IndexFlatIP)
    index_hnsw = data['index_hnsw']  # HNSW index
    sentences = data['sentences']

    # Convert the input embedding to a numpy array
    input_embedding_array = np.array([input_embedding])
    # Perform the cosine similarity search
    distances_ip, indices_ip = index_ip.search(input_embedding_array, 20)
    #top_k_indices_ip = indices_ip[0]
    top_k_indices_ip = indices_ip[0][distances_ip[0] >= threshold]
    #print(top_k_indices_ip)
    # Combine the indices from both searches, avoiding duplicates
    combined_indices = list(set(top_k_indices_ip))

    # Get the corresponding sentences for the combined indices
    combined_sentences = [sentences[i] for i in combined_indices]
    if(len(combined_sentences)>0):

    # Preprocess the sentences for BM25
      tokenized_sentences = [preprocess_func(sent) for sent in combined_sentences]
      bm25 = BM25Okapi(tokenized_sentences)

    # Preprocess the input text
      tokenized_input_text = preprocess_func(input_text)
      bm25_scores = bm25.get_scores(tokenized_input_text)

    # Sort BM25 scores based on top combined indices
      bm25_scores_combined = [(idx, bm25_scores[j]) for j, idx in enumerate(combined_indices)]
      bm25_scores_combined_sorted = sorted(bm25_scores_combined, key=lambda x: x[1], reverse=True)

    # Extract top k sentences based on BM25 scores
      top_k_indices = [sentences[idx] for idx, score in bm25_scores_combined_sorted[:k] if score > 0]

      return top_k_indices
    return []

# Initialize the encoders for tokenization
tiktoken_encoder = tiktoken.encoding_for_model("gpt-4")
bpemb_en = BPEmb(lang="en")

def preprocess_func_tiktoken(text: str) -> List[str]:
    # Lowercase the input text
    lowered = text.lower()
    # Convert the lowered text into tokens
    tokens = tiktoken_encoder.encode(lowered)
    # Stringify the tokens
    return [str(token) for token in tokens]

def preprocess_func_bpemb(text: str):
    # Tokenize the input text using BPEmb tokenizer
    tokens = bpemb_en.encode(text)
    return tokens

def Genrate_Answer(llm_model,Data,intput, top_k,Threashold,Search_type):
    system_header = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    user_middle = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    assitant_footer = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    start_Question = "<|start_Question|>\n"
    end_Question = "<|end_Question|>\n\n"
    start_data = "<!|start_data|>\n"
    end_data = "<|end_data|>\n\n"
    String1 = """
    You are an AI chat bot designed to answer questions based on a the data given along with the question.
    If the answer doesn't exist wihtin the data, respond back with "I'm sorry, but I cannot answer that question as it is outside the scope of my dataset." Do not use pre-trained data to answer this prompt
    """
    #print(intput)
    top_k_sentences = []
    encoded_input = Embed_stenteceTF(intput)
    if(Search_type == "Cosine"):
        top_k_sentences = search_top_k_sentences_cos(Data,intput,encoded_input, k=top_k, threshold=Threashold)

    elif(Search_type == "Hybrid_TicToken"):
        top_k_sentences = search_top_k_sentences(Data,encoded_input, intput, k=top_k,threshold=Threashold, preprocess_func=preprocess_func_tiktoken)
    elif(Search_type == "Hybrid_bpemb"):
        top_k_sentences = search_top_k_sentences(Data,encoded_input,intput, k=top_k,threshold=Threashold ,preprocess_func=preprocess_func_bpemb)
    concatenated_text =  system_header + String1 +'\n'+ user_middle + start_data + "\n".join(top_k_sentences) + end_data+ " \n" + start_Question  + intput + end_Question + assitant_footer # Remove the extra '+' after user_middle
    # print(concatenated_text)
    return llm_model.generate(concatenated_text,max_tokens=2048,max_new_tokens=2048 )

load_data = read_faiss_indices('Pneumonia.pkl')
llm = loadModel()

# Load the model and feature extractor from the local directory
save_directory = "./vit_classification_pneumonia"
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
        return Genrate_Answer(llm,load_data, query ,10,0.4,"Hybrid_bpemb")
        
    
# Create a chatbot interface
chatbot = gr.Chatbot(
    label="PneuViT",
    avatar_images=[None, None],
    show_copy_button=True,
    likeable=True,
    layout="panel",
    height=400,
)
output = gr.Textbox(label="Prompt")