import os
import io
import time
import pickle
import random
import numpy as np
from PIL import Image

import streamlit as st
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

st.set_page_config(
    page_title="TrendMatch · AI Fashion", 
    page_icon="✦", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

def init_session():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_results" not in st.session_state:
        st.session_state.last_results = None
    if "last_image" not in st.session_state:
        st.session_state.last_image = None
    if "last_name" not in st.session_state:
        st.session_state.last_name = None
    if "last_colors" not in st.session_state:
        st.session_state.last_colors = []
    if "last_url" not in st.session_state:
        st.session_state.last_url = None
    if "last_time" not in st.session_state:
        st.session_state.last_time = 0.0

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500;600&display=swap');

:root {
  --bg: #0A0A0A; 
  --surface: #141414; 
  --surface-2: #1C1C1C; 
  --accent: #FFFFFF;
  --accent-light: #E0E0E0; 
  --accent-pale: #2A2A2A; 
  --gold: #FFFFFF; 
  --gold-light: #A0A0A0;
  --ink: #FFFFFF; 
  --ink-2: #D0D0D0; 
  --muted: #888888; 
  --border: #2A2A2A;
  --border-2: #3A3A3A; 
  --shadow: rgba(0,0,0,0.80);
}

html, body, [class*="css"] { 
    font-family: 'Inter', sans-serif; 
    background-color: var(--bg) !important; 
    color: var(--ink); 
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stAppViewContainer"] { background: var(--bg) !important; }
.app-shell { padding: 0 3rem 4rem 3rem; max-width: 1360px; margin: 0 auto; }

.topnav { 
    display: flex; align-items: center; justify-content: space-between; 
    padding: 1.5rem 3rem; border-bottom: 1px solid var(--border); 
    background: var(--surface); position: sticky; top: 0; 
    z-index: 100; backdrop-filter: blur(8px); 
}
.topnav-brand { 
    font-family: 'Playfair Display', serif; font-size: 1.8rem; 
    font-weight: 400; color: var(--ink); letter-spacing: 0.15em; 
    text-transform: uppercase; 
}
.topnav-brand span { color: var(--muted); }
.topnav-sub { font-size: 0.58rem; letter-spacing: 0.28em; text-transform: uppercase; color: var(--muted); }

.page-hero { padding: 3rem 0 2rem 0; display: flex; align-items: flex-end; justify-content: space-between; border-bottom: 1px solid var(--border); margin-bottom: 2.5rem; }
.hero-title { font-family: 'Playfair Display', serif; font-size: 3.5rem; font-weight: 400; line-height: 1.1; color: var(--ink); }
.hero-title em { font-style: italic; color: var(--muted); }
.hero-desc { font-size: 0.72rem; color: var(--muted); letter-spacing: 0.06em; line-height: 1.7; max-width: 340px; text-align: right; }

.stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 0px; overflow: hidden; margin-bottom: 2.5rem; }
.stat-card { background: var(--surface); padding: 1.5rem 1.75rem; position: relative; }
.stat-card::after { content: ''; position: absolute; bottom: 0; left: 1.75rem; right: 1.75rem; height: 2px; background: transparent; transition: background 0.3s; }
.stat-card:hover::after { background: var(--accent); }
.stat-value { font-family: 'Playfair Display', serif; font-size: 2.4rem; font-weight: 400; line-height: 1; color: var(--ink); }
.stat-value.ok { color: var(--accent); }
.stat-value.warn { color: #ff4b4b; }
.stat-label { font-size: 0.6rem; letter-spacing: 0.22em; text-transform: uppercase; color: var(--muted); margin-top: 0.35rem; }

.stTabs [data-baseweb="tab-list"] { gap: 0 !important; background: transparent !important; border-bottom: 1px solid var(--border) !important; margin-bottom: 2rem; }
.stTabs [data-baseweb="tab"] { font-family: 'Inter', sans-serif !important; font-size: 0.65rem !important; letter-spacing: 0.22em !important; text-transform: uppercase !important; color: var(--muted) !important; background: transparent !important; border: none !important; padding: 0.75rem 1.75rem !important; border-bottom: 2px solid transparent !important; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }

.s-label { font-size: 0.6rem; letter-spacing: 0.25em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.6rem; }
.s-label::after { content:''; flex:1; height:1px; background:var(--border); }

.upload-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 0px; padding: 2rem; }
[data-testid="stFileUploader"] { border: 1px dashed var(--border-2) !important; border-radius: 0px !important; background: var(--surface-2) !important; transition: border-color 0.2s; }
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }
[data-testid="stFileUploader"] label { font-size: 0.7rem !important; letter-spacing: 0.1em !important; color: var(--ink) !important; }

.stButton > button { background: #1C1C1C !important; border: 1px solid #3A3A3A !important; border-radius: 0px !important; padding: 0.75rem 2.5rem !important; width: 100% !important; transition: all 0.2s !important; }
.stButton > button * { color: #FFFFFF !important; font-family: 'Inter', sans-serif !important; font-size: 0.7rem !important; font-weight: 600 !important; letter-spacing: 0.22em !important; text-transform: uppercase !important; transition: color 0.2s !important; }
.stButton > button:hover { background: #FFFFFF !important; border-color: #FFFFFF !important; transform: translateY(-1px); }
.stButton > button:hover * { color: #000000 !important; }
.stButton > button:active { transform: translateY(0) !important; }

.rec-outer { background: var(--surface); border: 1px solid var(--border); border-radius: 0px; overflow: hidden; transition: all 0.25s ease; margin-bottom: 1rem; }
.rec-outer:hover { border-color: var(--border-2); box-shadow: 0 8px 28px var(--shadow); transform: translateY(-3px); }
.rec-footer { display: flex; align-items: center; justify-content: space-between; padding: 0.6rem 0.9rem; border-top: 1px solid var(--border); }
.rec-rank { font-family: 'Playfair Display', serif; font-size: 1.1rem; color: var(--muted); }
.rec-sim { font-size: 0.6rem; letter-spacing: 0.12em; text-transform: uppercase; color: #000; background: var(--accent-light); padding: 3px 8px; border-radius: 0px; }
.rec-sim.high { background: var(--accent); color: #000; }

.empty-state { background: var(--surface-2); border: 1px dashed var(--border-2); border-radius: 0px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 1rem; padding: 5rem 2rem; text-align: center; }
.empty-icon { font-size: 3rem; opacity: 0.4; color: var(--muted); }
.empty-text { font-size: 0.65rem; letter-spacing: 0.22em; text-transform: uppercase; color: var(--muted); }

[data-testid="stSidebar"] { background: var(--surface-2) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color: var(--ink) !important; }
[data-testid="stSidebar"] label { font-size: 0.62rem !important; letter-spacing: 0.18em !important; text-transform: uppercase !important; color: var(--muted) !important; font-weight: 500 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-family: 'Playfair Display', serif !important; font-weight: 400 !important; }
[data-testid="stSidebar"] .stSelectbox > div > div { background: var(--surface) !important; border-color: var(--border-2) !important; border-radius: 0px !important; }
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] { background: var(--accent) !important; }
[data-testid="stSidebar"] .stSlider div[data-baseweb="slider"] > div:first-child { background: var(--border-2) !important; }
.stProgress > div > div { background: var(--accent) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-baseweb="toggle"] [data-checked="true"] { background: var(--accent) !important; }
[data-baseweb="toggle"] [data-checked="true"] > div { background: #000000 !important; }

.stTextInput input { border-radius: 0px !important; border-color: var(--border-2) !important; font-family: 'Inter', sans-serif !important; font-size: 0.85rem !important; background: var(--surface) !important; color: var(--ink) !important; }
.stTextInput input:focus { border-color: var(--accent) !important; box-shadow: none !important; }

.hist-card { background: var(--surface); border: 1px solid var(--border); border-radius: 0px; overflow: hidden; transition: box-shadow 0.2s; }
.hist-card:hover { box-shadow: 0 6px 20px var(--shadow); }
.hist-meta { padding: 0.5rem 0.75rem; font-size: 0.62rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); border-top: 1px solid var(--border); }

.chip { display: inline-block; background: #1A1A1A; border: 1px solid #333333; border-radius: 0px; font-size: 0.62rem; letter-spacing: 0.12em; text-transform: uppercase; padding: 3px 12px; margin: 3px; color: #FFFFFF; }
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }
code { background: #2A2A2A !important; color: #FFFFFF !important; border-radius: 0px !important; }
pre { background: var(--surface-2) !important; border: 1px solid var(--border) !important; border-radius: 0px !important; }
.color-swatch { display: inline-block; width: 24px; height: 24px; border-radius: 50%; border: 1px solid var(--border); margin-right: 6px; box-shadow: 0 2px 5px rgba(0,0,0,0.5); }
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_vision_model():
    try:
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
        from tensorflow.keras.preprocessing import image as keras_img
        from tensorflow.keras.layers import GlobalMaxPooling2D
        from tensorflow.keras.models import Sequential
        
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        
        feature_extractor = Sequential([
            base_model,
            GlobalMaxPooling2D()
        ])
        return feature_extractor, preprocess_input, keras_img
    except ImportError:
        return None, None, None

@st.cache_data(show_spinner=False)
def get_catalog_data(embeddings_path="embeddings.pkl", filenames_path="filenames.pkl"):
    if not os.path.exists(embeddings_path) or not os.path.exists(filenames_path):
        return None, None
        
    with open(embeddings_path, "rb") as f:
        embeddings = pickle.load(f)
        
    with open(filenames_path, "rb") as f:
        file_list = pickle.load(f)
        
    return np.array(embeddings), file_list

def process_image_features(img, model, preprocess, keras_img):
    resized_img = img.convert("RGB").resize((224, 224))
    img_array = keras_img.img_to_array(resized_img)
    expanded_img = np.expand_dims(img_array, 0)
    preprocessed_img = preprocess(expanded_img)
    
    features = model.predict(preprocessed_img, verbose=0).flatten()
    norm = np.linalg.norm(features)
    
    if norm > 0:
        return features / norm
    return features

def find_matches(query_features, catalog_matrix, num_results=6):
    knn = NearestNeighbors(n_neighbors=num_results+1, algorithm="brute", metric="cosine")
    knn.fit(catalog_matrix)
    
    distances, indices = knn.kneighbors([query_features])
    return indices[0][1:num_results+1], distances[0][1:num_results+1]

def extract_palette(image, num_colors=3):
    small_img = image.copy().resize((50, 50)).convert("RGB")
    pixel_data = np.array(small_img).reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixel_data)
    
    rgb_colors = kmeans.cluster_centers_.astype(int)
    hex_colors = []
    
    for color in rgb_colors:
        hex_val = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
        hex_colors.append(hex_val)
        
    return hex_colors

def main():
    init_session()

    with st.spinner("Initializing platform..."):
        vision_model, preprocess_fn, keras_img = get_vision_model()
        catalog_embeddings, image_paths = get_catalog_data()

    is_model_ready = vision_model is not None
    is_catalog_ready = catalog_embeddings is not None

    def execute_search(img_obj, img_filename=""):
        if not is_model_ready:
            st.error("Engine missing. Please install TensorFlow.")
            return
            
        if not is_catalog_ready:
            st.error("Catalog data not found. Run feature extraction first.")
            return

        with st.spinner("Processing visual features..."):
            start_time = time.time()
            
            features = process_image_features(img_obj, vision_model, preprocess_fn, keras_img)
            match_indices, match_distances = find_matches(features, catalog_embeddings, max_results)
            
            end_time = time.time()
            
            st.session_state.last_time = end_time - start_time
            st.session_state.last_results = (match_indices, match_distances)
            st.session_state.last_image = img_obj
            st.session_state.last_name = img_filename if img_filename else "Uploaded Image"
            
            history_record = {
                "name": st.session_state.last_name,
                "image": img_obj.copy(),
                "n": len(match_indices)
            }
            st.session_state.history.append(history_record)
            
            if enable_palette:
                st.session_state.last_colors = extract_palette(img_obj)
            else:
                st.session_state.last_colors = []

    with st.sidebar:
        st.markdown("## TrendMatch AI")
        st.caption("Fashion Intelligence Platform")
        st.markdown("---")
        
        st.markdown("### Search Parameters")
        max_results = st.slider("Max Results", 3, 12, 6)
        grid_layout = st.select_slider("Grid Size", [2, 3, 4], value=3)
        
        show_metrics = st.toggle("Show Match %", value=True)
        show_labels = st.toggle("Show Filenames", value=False)
        enable_palette = st.toggle("Color Analysis", value=True)
        
        st.markdown("---")
        st.markdown("### Quick Actions")
        
        if st.button("🎲 Surprise Me"):
            if is_catalog_ready and len(image_paths) > 0:
                random_idx = random.randint(0, len(image_paths) - 1)
                selected_image = image_paths[random_idx]
                
                try:
                    loaded_img = Image.open(selected_image)
                    st.session_state.last_url = None
                    execute_search(loaded_img, os.path.basename(selected_image))
                except Exception as ex:
                    st.error(f"Failed to load image: {str(ex)}")
            else:
                st.warning("Catalog is empty.")

    st.markdown("""
    <div class="topnav">
      <div>
        <div class="topnav-brand">TRENDMATCH<span> ✦</span></div>
        <div class="topnav-sub">Visual Discovery Engine</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="app-shell">', unsafe_allow_html=True)

    st.markdown("""
    <div class="page-hero">
      <div class="hero-title">Discover Your<br><em>Perfect Style</em></div>
      <div class="hero-desc">
        Upload any garment and our AI will surface visually similar pieces from the catalogue.
      </div>
    </div>
    """, unsafe_allow_html=True)

    total_items = len(image_paths) if is_catalog_ready else 0
    dims = catalog_embeddings.shape[1] if is_catalog_ready else 2048
    
    time_display = f"{st.session_state.last_time:.2f}s" if st.session_state.last_time > 0 else "-"
    status_class = "ok" if is_model_ready else "warn"
    
    st.markdown(f"""
    <div class="stats-row">
      <div class="stat-card">
        <div class="stat-value {status_class}" style="font-size: 1.6rem; padding-top: 0.5rem;">ResNet50</div>
        <div class="stat-label">AI Engine</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{total_items:,}</div>
        <div class="stat-label">Catalogue Items</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{dims:,}</div>
        <div class="stat-label">Feature Dimensions</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{time_display}</div>
        <div class="stat-label">Search Time</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab_search, tab_link, tab_history, tab_docs = st.tabs([
        "  Visual Search  ",
        "  Image URL  ",
        "  History  ",
        "  Documentation  "
    ])

    with tab_search:
        col_left, col_right = st.columns([1, 2], gap="large")

        with col_left:
            st.markdown('<p class="s-label">Your Garment</p>', unsafe_allow_html=True)
            
            upload_widget = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
            
            preview_img = None
            preview_name = ""
            is_new = False
            
            if upload_widget:
                preview_img = Image.open(upload_widget)
                preview_name = upload_widget.name
                if preview_name != st.session_state.last_name:
                    is_new = True
                    st.session_state.last_url = None
                    
            elif st.session_state.last_image and not st.session_state.last_url:
                preview_img = st.session_state.last_image
                preview_name = st.session_state.last_name
            
            if preview_img:
                st.image(preview_img, use_container_width=True)
                
                img_w, img_h = preview_img.size
                st.markdown(f'<p style="font-size:.6rem;letter-spacing:.1em;text-transform:uppercase;color:#888888;margin-top:.4rem;">{preview_name} &nbsp;·&nbsp; {img_w}×{img_h}</p>', unsafe_allow_html=True)
                
                if not is_new and st.session_state.last_colors:
                    palette_html = ""
                    for color_hex in st.session_state.last_colors:
                        palette_html += f'<span class="color-swatch" style="background-color: {color_hex};" title="{color_hex}"></span>'
                    st.markdown(f'<div style="margin-top: 0.5rem; display: flex; align-items: center;"><span style="font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;color:#888888;margin-right:8px;">Palette:</span>{palette_html}</div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if is_new:
                    if st.button("✦  Find Similar Pieces", use_container_width=True):
                        execute_search(preview_img, preview_name)
                        st.rerun()

        with col_right:
            st.markdown('<p class="s-label">Recommendations</p>', unsafe_allow_html=True)
            
            if not is_new and st.session_state.last_results and is_catalog_ready and not st.session_state.last_url:
                match_indices, match_distances = st.session_state.last_results
                grid_columns = st.columns(grid_layout)
                
                for i, (idx, dist) in enumerate(zip(match_indices, match_distances)):
                    match_percent = (1 - dist) * 100
                    highlight_class = "high" if match_percent >= 80 else ""
                    
                    file_path = image_paths[idx]
                    current_col = grid_columns[i % grid_layout]
                    
                    with current_col:
                        try:
                            result_img = Image.open(file_path)
                            st.image(result_img, use_container_width=True)
                        except FileNotFoundError:
                            placeholder = np.full((300, 300, 3), 240, dtype=np.uint8)
                            st.image(placeholder, use_container_width=True)
                            
                        display_label = os.path.basename(file_path) if show_labels else f"Style {i+1:02d}"
                        score_badge = f'<span class="rec-sim {highlight_class}">{match_percent:.1f}%</span>' if show_metrics else ""
                        
                        st.markdown(f'<div class="rec-footer"><span class="rec-rank">#{i+1:02d}</span><span style="font-size:.65rem;color:#888888;letter-spacing:.08em;">{display_label}</span>{score_badge}</div>', unsafe_allow_html=True)
                        st.markdown('<div style="display:flex; justify-content:space-between; padding:0 0.9rem 0.6rem; font-size:0.8rem;"><a href="#" style="text-decoration:none; color:#888888;" title="Save">🤍</a><a href="#" style="text-decoration:none; color:#888888;" title="Find similar">🔍</a></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="empty-state"><div class="empty-icon">✦</div><div class="empty-text">Upload a garment to discover similar pieces</div></div>', unsafe_allow_html=True)

    with tab_link:
        st.markdown('<p class="s-label">Paste Image URL</p>', unsafe_allow_html=True)
        link_input = st.text_input("", placeholder="https://example.com/garment.jpg", label_visibility="collapsed")
        
        if link_input:
            try:
                import requests
                response = requests.get(link_input, timeout=8)
                link_img = Image.open(io.BytesIO(response.content))
                
                is_new_link = (link_input != st.session_state.last_url)
                
                col_img, col_results = st.columns([1, 2], gap="large")
                
                with col_img:
                    st.image(link_img, use_container_width=True)
                    if is_new_link:
                        if st.button("✦  Find Similar", use_container_width=True):
                            st.session_state.last_url = link_input
                            execute_search(link_img, "URL Image")
                            st.rerun()
                    elif st.session_state.last_colors:
                        palette_html = ""
                        for color_hex in st.session_state.last_colors:
                            palette_html += f'<span class="color-swatch" style="background-color: {color_hex};"></span>'
                        st.markdown(f'<div style="margin-top: 0.5rem; display: flex; align-items: center;">{palette_html}</div>', unsafe_allow_html=True)
                        
                with col_results:
                    if not is_new_link and st.session_state.last_results and is_catalog_ready:
                        match_indices, match_distances = st.session_state.last_results
                        res_cols = st.columns(grid_layout)
                        
                        for i, (idx, dist) in enumerate(zip(match_indices, match_distances)):
                            with res_cols[i % grid_layout]:
                                try:
                                    st.image(Image.open(image_paths[idx]), use_container_width=True)
                                except Exception:
                                    pass
                                if show_metrics:
                                    st.caption(f"{(1-dist)*100:.1f}% match")
            except Exception as ex:
                st.error(f"Failed to load URL: {str(ex)}")

    with tab_history:
        st.markdown('<p class="s-label">Recent Searches</p>', unsafe_allow_html=True)
        
        if not st.session_state.history:
            st.markdown('<div class="empty-state" style="padding:3rem;"><div class="empty-icon">🕐</div><div class="empty-text">No previous searches</div></div>', unsafe_allow_html=True)
        else:
            recent_searches = st.session_state.history[-10:]
            hist_columns = st.columns(min(len(recent_searches), 5))
            
            for i, record in enumerate(reversed(recent_searches)):
                with hist_columns[i % len(hist_columns)]:
                    st.image(record["image"], use_container_width=True)
                    st.markdown(f'<div class="hist-meta">{record["name"]}<br>{record["n"]} results</div>', unsafe_allow_html=True)
                    
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Clear History"):
                st.session_state.history = []
                st.session_state.last_results = None
                st.session_state.last_image = None
                st.session_state.last_url = None
                st.rerun()

    with tab_docs:
        doc_col1, doc_col2 = st.columns(2, gap="large")
        
        with doc_col1:
            st.markdown("### System Architecture")
            st.markdown("The platform leverages a custom-tuned ResNet50 convolutional neural network. We drop the final classification layer in favor of Global Max Pooling to extract a 2048-dimensional feature vector per image.")
            st.markdown("For inference, the Euclidean distance is optimized using Cosine Similarity metrics across the vectorized catalog dataset via the K-Nearest Neighbors algorithm.")
            
        with doc_col2:
            st.markdown("### Environment Setup")
            st.code("pip install tensorflow scikit-learn streamlit pillow requests", language="bash")
            st.code("python feature_extractor.py\nstreamlit run app.py", language="bash")
            
        st.markdown("---")
        tags = ["ResNet50", "Computer Vision", "K-NN", "Embeddings", "Streamlit", "K-Means"]
        for tag in tags:
            st.markdown(f'<span class="chip">{tag}</span>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="border-top:1px solid #2A2A2A;padding:1.25rem 3rem; display:flex;justify-content:space-between;align-items:center; font-size:.6rem;letter-spacing:.14em;text-transform:uppercase; color:#888888;margin-top:3rem;">
      <span>TRENDMATCH ENGINE</span>
      <span>V1.0</span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()