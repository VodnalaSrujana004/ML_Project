import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Load Models
@st.cache_resource
def load_models():
    image_caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_caption_model.to(device)
    
    text_to_image_model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
    
    return processor, image_caption_model, text_to_image_model

processor, image_caption_model, text_to_image_model = load_models()

# Streamlit UI
st.title("AI Image Generator & Captioning App")
st.sidebar.write("Choose a function:")

option = st.sidebar.radio("Select Task", ["Text-to-Image Generation", "Image Captioning"])

if option == "Text-to-Image Generation":
    st.header("Text-to-Image Generation")
    prompt = st.text_input("Enter a prompt to generate an image:")
    
    if st.button("Generate Image"):
        with st.spinner("Generating..."):
            image = text_to_image_model(prompt).images[0]
            st.image(image, caption="Generated Image", use_column_width=True)

elif option == "Image Captioning":
    st.header("Image Captioning")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Generate Caption"):
            with st.spinner("Processing..."):
                inputs = processor(image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
                out = image_caption_model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)
                st.success(f"Caption: {caption}")
