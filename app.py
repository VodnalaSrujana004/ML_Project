import streamlit as st
from PIL import Image
import io
import base64
import torch
from diffusers import StableDiffusionPipeline
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="Image Generation and Captioning",
    page_icon=":camera:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Caching the model loading for better performance
@st.cache_resource
def load_stable_diffusion_pipeline():
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            use_auth_token=None # Replace None with your actual Hugging Face token if needed
        )
        pipe = pipe.to("cuda")  # Use GPU if available
        print("Stable Diffusion Pipeline loaded.")
        return pipe
    except Exception as e:
        print(f"Error loading Stable Diffusion: {e}")
        st.error(f"Error loading Stable Diffusion: {e}")  # Display error in Streamlit
        return None

@st.cache_resource
def load_image_to_text_pipeline():
    try:
        image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        print("Image to Text Pipeline loaded")
        return image_to_text
    except Exception as e:
        print(f"Error loading Image-to-Text: {e}")
        st.error(f"Error loading Image-to-Text: {e}")  # Display error in Streamlit
        return None


pipe = load_stable_diffusion_pipeline()
image_to_text = load_image_to_text_pipeline()


# Sidebar for options
st.sidebar.header("Options")
app_mode = st.sidebar.selectbox("Choose the App mode",
                                ["Image Generation", "Image Captioning"])


# Main Application
if app_mode == "Image Generation":
    st.title("Image Generation")
    prompt = st.text_input("Enter a prompt for image generation:")

    if st.button("Generate Image"):
        if pipe is None:
            st.error("Stable Diffusion pipeline not loaded.")
        elif not prompt:
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Generating image..."):
                try:
                    image = pipe(prompt).images[0]
                    st.image(image, caption=f"Generated image based on prompt: '{prompt}'", use_column_width=True)
                except Exception as e:
                    st.error(f"Error during image generation: {e}")

elif app_mode == "Image Captioning":
    st.title("Image Captioning")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Generate Caption"):
            if image_to_text is None:
                st.error("Image-to-text pipeline not loaded.")
            else:
                with st.spinner("Generating caption..."):
                    try:
                        caption = image_to_text(image)[0]["generated_text"]
                        st.write(f"**Caption:** *{caption}*")
                    except Exception as e:
                        st.error(f"Error during image captioning: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created with Streamlit and Hugging Face Transformers")
