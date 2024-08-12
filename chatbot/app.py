import gradio as gr

# Import modules from other files
from qdrant_embeddings_rag_chatbot import model_inference, chatbot

# Chat interface block
with gr.Blocks(
        css="""
        .gradio-container .avatar-container {height: 40px; width: 40px !important;}
        #duplicate-button {margin: auto; color: white; background: #f1a139; border-radius: 100vh; margin-top: 2px; margin-bottom: 2px;}
        """,
) as chat:
    # gr.Markdown("### Chatbot, Pneumonia classification and Normal Chat")
    gr.ChatInterface(
        fn=model_inference,
        chatbot = chatbot,
        # examples=EXAMPLES,
        multimodal=True,
        # cache_examples=False,
        autofocus=True,
        concurrency_limit=10,
    )
   
# Main application block
with gr.Blocks() as demo:
    gr.TabbedInterface([chat], ['💬 SuperChat'])

demo.queue(max_size=300)
demo.launch(share=True)
