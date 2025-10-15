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

st.set_page_config(page_title="Visual Product Matcher", layout="wide")

# --- CSS for animations and styling ---
st.markdown(
    """
    <style>
    .card {
      border-radius: 12px;
      padding: 10px;
      box-shadow: 0 6px 18px rgba(0,0,0,0.12);
      transition: transform .25s ease, box-shadow .25s ease;
      background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(250,250,250,0.9));
    }
    .card:hover {
      transform: translateY(-8px);
      box-shadow: 0 12px 30px rgba(0,0,0,0.18);
    }
    .match-img {
      width: 100%;
      height: 200px;
      object-fit: contain;
      border-radius: 8px;
      background: #fff;
      padding: 8px;
    }
    .title {
      font-size:20px;
      font-weight:700;
      margin-bottom:6px;
    }
    .subtext {
      color: #666;
      font-size:13px;
    }
    .animated-header {
      font-size:28px;
      font-weight:800;
      letter-spacing:1px;
      background: linear-gradient(90deg,#ff7a18,#af002d 50%,#319197 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: hue 6s infinite linear;
    }
    @keyframes hue {
      0% { filter: hue-rotate(0deg); }
      100% { filter: hue-rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    "<div style='display:flex;align-items:center;gap:18px'>"
    "<div class='animated-header'>Visual Product Matcher</div>"
    "<div style='color:#666'>— find visually similar items in your catalog</div>"
    "</div>",
    unsafe_allow_html=True
)
st.write("")

# Sidebar controls
st.sidebar.header("Controls")
K = st.sidebar.slider("Number of results", min_value=1, max_value=10, value=5)
use_gpu = st.sidebar.checkbox("Use GPU if available", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown("Tip: upload an image or provide an image URL to see similar products.")

# --- helper for base64 embedding in markdown ---
def b64_encode(b):
    return base64.b64encode(b).decode('utf-8')

# Load matcher (embeddings must already be computed)
@st.cache_resource
def load_matcher(use_gpu=False):
    device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
    matcher = VisualMatcher(embeddings_path='embeddings.npy', metadata_path='metadata.json')
    # load backbone for query embeddings
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
        "Could not load embeddings. Make sure you ran `extract_embeddings.py` "
        "and that embeddings.npy and metadata.json are in the working directory. "
        "Error: " + str(e)
    )
    st.stop()

# Main area - input selection
col1, col2 = st.columns([1,2])
with col1:
    st.markdown("### Input image")

    # Upload image
    uploaded = st.file_uploader("Upload a product image", type=['jpg','jpeg','png','webp'])
    input_image = None
    if uploaded is not None:
        input_image = Image.open(uploaded).convert('RGB')

    # Or use URL
    url_input = st.text_input("Or enter image URL")
    if url_input:
        try:
            img_bytes = urlopen(url_input).read()
            input_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            st.error(f"Could not load image from URL: {e}")

    if input_image is not None:
        st.image(input_image, caption="Query image", use_container_width=True)

with col2:
    st.markdown("### Results")
    if input_image is None:
        st.info("Upload or enter a URL to see matches.")
    else:
        # compute embedding
        img_t = transform(input_image).unsqueeze(0).to(device)
        with torch.no_grad():
            q_emb = model(img_t).cpu().numpy().reshape(-1)
        results = matcher.query(q_emb, topk=K)

        ncols = 3
        cols = st.columns(ncols)
        for i, res in enumerate(results):
            col = cols[i % ncols]
            file = res['file']
            score = res['score']
            try:
                # fix path for relative metadata paths
                img = Image.open(file).convert('RGB')



                buf = io.BytesIO()
                img.save(buf, format='JPEG')
                b = buf.getvalue()
                with col:
                    st.markdown(
                        f"<div class='card'>"
                        f"<img class='match-img' src='data:image/jpeg;base64,{b64_encode(b)}' />"
                        f"<div style='padding:6px;'>"
                        f"<div class='title'>Match #{i+1}</div>"
                        f"<div class='subtext'>Score: {score:.4f}</div>"
                        f"</div></div>",
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.write(f"Could not load {file}: {e}")
