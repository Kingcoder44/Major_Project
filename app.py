import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torchvision import transforms, models

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Medgic - AI Skin Disease Detection",
    page_icon="🔬",
    layout="centered"
)

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1E88E5;
        font-size: 3rem;
        font-weight: bold;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
    }
    .disclaimer {
        background-color: #FFF3E0;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #FF9800;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Load Model
# ---------------------------


@st.cache_resource
def load_model():
    num_classes = 22

    model = EfficientNet.from_name('efficientnet-b0')
    model._fc = torch.nn.Linear(model._fc.in_features, num_classes)

    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()

    return model

# ---------------------------
# Load Labels
# ---------------------------
@st.cache_data
def load_labels():
    with open("labels.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

# ---------------------------
# Image Preprocessing
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

# ---------------------------
# Prediction Function
# ---------------------------
def predict(image, model, labels):
    img_tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs[0], dim=0)

    top_probs, top_indices = torch.topk(probabilities, 5)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "disease": labels[idx].replace("_", " "),
            "confidence": float(prob)
        })

    return results

# ---------------------------
# Main App
# ---------------------------
def main():
    st.markdown('<p class="main-header">🔬 Medgic</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Skin Disease Detection</p>', unsafe_allow_html=True)

    st.markdown("""
        <div class="disclaimer">
            <strong>⚠️ MEDICAL DISCLAIMER</strong><br>
            This application is for educational purposes only.
            It is NOT a substitute for professional medical diagnosis.
            Always consult a qualified dermatologist.
        </div>
    """, unsafe_allow_html=True)

    try:
        model = load_model()
        labels = load_labels()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

    st.markdown("### 📸 Upload Skin Image")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image")

        with col2:
            if st.button("🔬 Analyze Skin"):
                with st.spinner("Analyzing..."):
                    results = predict(image, model, labels)

                st.markdown("#### Top 5 Predictions")
                for i, result in enumerate(results):
                    confidence_pct = result["confidence"] * 100
                    st.write(f"{i+1}. {result['disease']} — {confidence_pct:.2f}%")
                    st.progress(result["confidence"])

    st.markdown("---")
    st.markdown("Developed by Kushagra Srivastava | Final Year Project")

# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    main()