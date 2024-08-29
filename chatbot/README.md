# Run the PneumonoBot Locally

## Clone the Repository

```bash
git clone https://github.com/Muhammad-Yaqoob-4110/PneumonoBot.git
```
## Load the Model from Hugging Face
To run the PneumonoBot locally, first, load the model from Hugging Face:
```python
from transformers import ViTForImageClassification, ViTFeatureExtractor

# Load the model and feature extractor from Hugging Face
model_name = "M-Yaqoob/PneumonoBot"
model = ViTForImageClassification.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# Save the model and feature extractor locally
save_directory = "./vit_classification_pneumonobot"
model.save_pretrained(save_directory)
feature_extractor.save_pretrained(save_directory)

```
## Install Dependencies
Navigate to the chatbot directory and install the required dependencies:
```bash
cd chatbot
pip install -r requirements.txt
```

## Configuration
Create a configuration file named config.json in the PneumonoBot/chatbot directory with the following content:
```json
{
  "qdrant_url": "QDRANT_URL",
  "qdrant_api_key": "QDRANT_API_KEY",
  "nvidia_api_key": "NVIDIA_API_KEY",
  "lamini_api_key": "LAMINI_API_KEY",
  "gemini_api_key": "GEMINI_API_KEY"
}
```

- To get the Nvidia API key, follow this [Nvidia API Key](https://org.ngc.nvidia.com/setup/api-key).
- To get the Qdrant URL and API key, follow the official documentation: [Qdrant docs](https://qdrant.tech/documentation/qdrant-cloud-api/).
- To get the Lamini API key, follow this [Lamini API Key](https://lamini-ai.github.io/authenticate/).
- You can get your free Google Gemini API key by following this: [Get your Google Gemini API Key](https://ai.google.dev/gemini-api/docs/api-key).

## Running the Application
To run the Gradio application, execute:
```bash
gradio app.py
```
This will start the server. Feel free to try out PneumonoBot by uploading X-ray images and asking questions related to pneumonia through the chatbot.