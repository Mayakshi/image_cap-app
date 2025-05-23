import streamlit as st
import torch
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np

# --- Load Models (dummy structure, replace with your actual model loading) ---
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
from utils.vocabulary import Vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load vocabulary
vocab = Vocabulary.load('models/vocab.pkl')

# Load captioning models
encoder = EncoderCNN(embed_size=256).to(device)
encoder.load_state_dict(torch.load('models/encoder.pth', map_location=device))
encoder.eval()
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(vocab)).to(device)
decoder.load_state_dict(torch.load('models/decoder.pth', map_location=device))
decoder.eval()

# Load segmentation model
import torchvision
seg_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
seg_model.eval()

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def get_caption(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = encoder(image_tensor)
        sampled_ids = decoder.sample(features)
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == '<end>':
            break
        sampled_caption.append(word)
    return ' '.join(sampled_caption)

def get_segmentation(image):
    image_np = np.array(image.convert('RGB'))
    image_tensor = transforms.ToTensor()(image_np).unsqueeze(0)
    with torch.no_grad():
        prediction = seg_model(image_tensor)[0]
    # Overlay masks for visualization
    masks = prediction['masks'] > 0.5
    for mask in masks:
        mask = mask.squeeze().cpu().numpy()
        image_np[mask] = [255, 0, 0]  # Red overlay
    return Image.fromarray(image_np)

# --- Streamlit UI ---
st.title("Image Captioning & Segmentation Demo")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Generating caption and segmentation...")

    caption = get_caption(image)
    st.markdown(f"**Caption:** {caption}")

    segmented_image = get_segmentation(image)
    st.image(segmented_image, caption='Segmented Image', use_column_width=True)
