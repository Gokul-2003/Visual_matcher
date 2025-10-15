import streamlit as st
from PIL import Image
import numpy as np
import io, os
import torch
import torchvision.transforms as T
import torchvision.models as models
from model import VisualMatcher
import base64
from urllib.request import urlopen

st.set_page_config(page_title="Visual Product Identifier and Matcher", layout="wide")

# --- Modern Glassmorphism UI ---
st.markdown("""
<style>
body {
  background: linear-gradient(135deg, #e0eafc, #cfdef3);
  font-family: "Poppins", sans-serif;
}
.main {
  padding: 2rem 3rem;
}
.header {
  text-align: center;
  margin-bottom: 1.5rem;
}
.header h1 {
  font-size: 2.3rem;
  background: linear-gradient(90deg,#00C9FF,#92FE9D);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 800;
  letter-spacing: 1px;
}
.header p {
  color: #444;
  font-size: 1rem;
}
.sidebar .sidebar-content {
  background: rgba(255,255,255,0.7);
  backdrop-filter: blur(12px);
  border-radius: 12px;
}
.stButton > button {
  background: linear-gradient(90deg,#00C9FF,#92FE9D);
  color: #fff;
  border: none;
  border-radius: 8px;
  padding: 0.5rem 1.2rem;
  font-weight: 600;
  transition: all 0.3s ease;
}
.stButton > button:hover {
  transform: scale(1.05);
  background: linear-gradient(90deg,#92FE9D,#00C9FF);
}
.card {
  border-radius: 15px;
  background: rgba(255,255,255,0.55);
  backdrop-filter: blur(10px);
  box-shadow: 0 8px 25px rgba(0,0,0,0.1);
  transition: all 0.25s ease;
  padding: 12px;
}
.card:hover {
  transform: translateY(-6px);
  box-shadow: 0 12px 35px rgba(0,0,0,0.15);
}
.match-img {
  width: 100%;
  height: 220px;
  border-radius: 10px;
  object-fit: cover;
}
.match-info {
  padding-top: 8px;
  text-align: center;
}
.match-info h4 {
  font-size: 17px;
  margin: 2px 0;
  font-weight: 700;
}
.match-info p {
  font-size: 13px;
  color: #666;
  margin: 0;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="header">
  <h1>🧠 Visual Product Matcher</h1>
  <p>Upload or paste an image URL to find visually similar products from your catalog.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Controls")
K = st.sidebar.slider("Number of results", min_value=1, max_value=10, value=5)
use_gpu = st.sidebar.checkbox("Use GPU if available", value=False)
st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Upload or paste an image URL to find similar products!")

# --- helper ---
def b64_encode(b):
    return base64.b64encode(b).decode('utf-8')

# --- Load Matcher ---
@st.cache_resource
def load_matcher(use_gpu=False):
    device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
    matcher = VisualMatcher(embeddings_path='embeddings.npy', metadata_path='metadata.json')
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    transform = T.Compose([
        T.Resize((224,224)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    return matcher, model, transform, device

try:
    matcher, model, transform, device = load_matcher(use_gpu=use_gpu)
except Exception as e:
    st.error(
        "⚠️ Could not load embeddings. Please run `extract_embeddings.py` and ensure files exist.\n\n"
        f"Error: {e}"
    )
    st.stop()

# --- Layout ---
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("📷 Input Image")

    uploaded = st.file_uploader("Upload image", type=['jpg','jpeg','png','webp'])
    input_image = None
    if uploaded is not None:
        input_image = Image.open(uploaded).convert('RGB')

    url_input = st.text_input("Or enter image URL")
    if url_input:
        try:
            img_bytes = urlopen(url_input).read()
            input_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            st.error(f"❌ Could not load image: {e}")

    if input_image is not None:
        st.image(input_image, caption="Query Image", use_container_width=True)

with col2:
    st.subheader("🔍 Matching Results")

    if input_image is None:
        st.info("Upload or enter an image URL to start matching.")
    else:
        img_t = transform(input_image).unsqueeze(0).to(device)
        with torch.no_grad():
            q_emb = model(img_t).cpu().numpy().reshape(-1)
        results = matcher.query(q_emb, topk=K)

        cols = st.columns(3)
        for i, res in enumerate(results):
            col = cols[i % 3]
            file = res['file']
            score = res['score']
            try:
                img = Image.open(file).convert('RGB')
                buf = io.BytesIO()
                img.save(buf, format='JPEG')
                b = buf.getvalue()

                with col:
                    st.markdown(
                        f"""
                        <div class='card'>
                            <img class='match-img' src='data:image/jpeg;base64,{b64_encode(b)}'>
                            <div class='match-info'>
                                <h4>Match #{i+1}</h4>
                                <p>Score: {score:.4f}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Could not load {file}: {e}")
