import streamlit as st
import tensorflow as tf
import numpy as np
import os
import json
import cv2
from PIL import Image
import warnings
import pillow_heif

# Register HEIF opener for Apple photos support (.heic)
try:
    pillow_heif.register_heif_opener()
except:
    pass

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & SLEEK CUSTOM STYLING (AESTHETICS FIRST!)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DEC Museum Artifact Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Curatedsleek HSL dark mode style
st.markdown("""
<style>
    /* Main App Background & Text */
    .stApp {
        background-color: #0E1117;
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Elegant Title Styling */
    .app-title {
        background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 50%, #1D4ED8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.05em;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Styled Containers & Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        margin-bottom: 1.5rem;
    }
    
    /* Custom Headers inside Cards */
    .card-header {
        color: #F8FAFC;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 0.5rem;
    }
    
    /* Metrics Highlighting */
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #60A5FA;
        margin-bottom: 0.1rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Custom Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Progress Bar Color */
    div[data-testid="stProgressBar"] > div > div {
        background-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CACHED MODEL LOADING & DYNAMIC CONFIGS
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "hasil_dec_softmax_10d", "model")
GRAFIK_DIR = os.path.join(BASE_DIR, "hasil_dec_softmax_10d", "grafik")
CLUSTER_DIR = os.path.join(BASE_DIR, "hasil_dec_softmax_10d", "cluster_dec")

CLASS_NAMES = [
    "koleksi arkeologi",
    "koleksi biologi",
    "koleksi etnografi",
    "koleksi geologi",
    "koleksi keramik",
    "koleksi kesenian",
    "koleksi naskah kuno",
    "koleksi numismatik",
    "koleksi senjata",
    "koleksi teknologika"
]

@tf.keras.utils.register_keras_serializable(package="Custom")
class DECLayer(tf.keras.layers.Layer):
    def __init__(self, n_clusters, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha      = alpha

    def build(self, input_shape):
        self.clusters = self.add_weight(
            shape=(self.n_clusters, input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True, name='cluster_centers'
        )
        super().build(input_shape)

    def call(self, inputs):
        sq_dist = tf.reduce_sum(
            tf.square(tf.expand_dims(inputs, 1) - self.clusters), axis=2
        )
        numerator = tf.pow(1.0 + sq_dist / self.alpha, -(self.alpha + 1.0) / 2.0)
        q = numerator / tf.reduce_sum(numerator, axis=1, keepdims=True)
        return q

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'n_clusters': self.n_clusters, 'alpha': self.alpha})
        return cfg

@st.cache_resource
def load_dec_models():
    """Load dec and enc models from disk once by dynamically reconstructing architecture."""
    dec_path = os.path.join(MODEL_DIR, "dec_model.keras")
    ft_model_path = os.path.join(BASE_DIR, "hasil_finetuned", "model", "finetuned_final.keras")
    
    # Resolved relative to BASE_DIR for universal portability
    
    # 1. Load fine-tuned backbone & head layers
    ft_model = tf.keras.models.load_model(ft_model_path, compile=False)
    
    backbone = ft_model.get_layer('efficientnetv2-b2')
    dense_512 = ft_model.get_layer('fc512')
    bn_layer  = ft_model.get_layer('bn1')
    dense_256 = ft_model.get_layer('fc256')
    classifier_layer = ft_model.get_layer('classifier')
    
    # Freeze backbone
    backbone.trainable = False
    
    # 2. Build functional models
    inp     = tf.keras.Input(shape=(224, 224, 3), name='input')
    enc_out = backbone(inp, training=False)
    z_512   = dense_512(enc_out)
    z_bn    = bn_layer(z_512)
    z_256   = dense_256(z_bn)
    z_softmax = classifier_layer(z_256)
    
    # Add L2 normalization and DECLayer
    z_norm  = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='l2_norm')(z_softmax)
    q       = DECLayer(10, alpha=1.0, name='dec_layer')(z_norm)
    
    dec = tf.keras.Model(inp, q, name='DEC')
    enc = tf.keras.Model(inp, z_norm, name='Encoder')
    
    # 3. Load DEC weights from disk (bypasses Lambda deserialization issues completely!)
    dec.load_weights(dec_path)
    
    return dec, enc

try:
    dec_model, enc_model = load_dec_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    error_msg = str(e)

# -----------------------------------------------------------------------------
# 3. CORE IMAGE PREPROCESSING FUNCTIONS
# -----------------------------------------------------------------------------
def preprocess_for_prediction(img_array, img_size=224):
    """
    Sleek, notebook-identical preprocessing pipeline:
    Resizes the PIL image to img_size x img_size using LANCZOS filter
    and preserves the raw float32 pixel values [0, 255] as expected by the model.
    """
    # 1. Resize exactly as in DEC_Best_Method.ipynb
    img_resized = img_array.resize((img_size, img_size), Image.LANCZOS)
    img_tensor = np.array(img_resized, dtype=np.float32)
    return img_tensor, img_resized

# -----------------------------------------------------------------------------
# 4. SIDEBAR - METRICS & ARCHITECTURE DUMP
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏛️ DEC Model Info")
    st.markdown("---")
    st.markdown(
        "Aplikasi web ini menggunakan arsitektur **Deep Embedded Clustering (DEC)** "
        "yang telah dioptimasi dengan inisialisasi **Identity Centroid** "
        "pada ruang laten **10D Softmax ter-L2-normalisasi**."
    )
    
    st.markdown("### 📊 Ringkasan Metrik Terbaik:")
    
    # Custom HTML cards for metrics to look highly polished
    st.markdown("""
    <div class="glass-card" style="padding:15px; margin-bottom:10px;">
        <div class="metric-value" style="font-size:1.8rem; color:#10B981;">0.9785</div>
        <div class="metric-label">Best Silhouette Score</div>
    </div>
    <div class="glass-card" style="padding:15px; margin-bottom:10px;">
        <div class="metric-value" style="font-size:1.8rem; color:#60A5FA;">0.0845</div>
        <div class="metric-label">Best Davies-Bouldin Index (DBI)</div>
    </div>
    <div class="glass-card" style="padding:15px; margin-bottom:10px;">
        <div class="metric-value" style="font-size:1.8rem; color:#F59E0B;">95.11%</div>
        <div class="metric-label">Clustering Accuracy</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("Dibuat untuk Pengujian Sidang Skripsi - ML Museum")

# -----------------------------------------------------------------------------
# 5. MAIN HEADER & DYNAMIC VIEWS
# -----------------------------------------------------------------------------
st.markdown('<div class="app-title">🏛️ Deep Embedded Clustering (DEC) Web App</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Dashboard Interaktif Pengelompokan & Analisis Citra Benda Arkeologi Museum</div>', unsafe_allow_html=True)

if not models_loaded:
    st.error(f"⚠️ Gagal memuat model DEC dari disk. Harap pastikan model sudah ditraining. Rincian error: {error_msg}")
    st.stop()

# Tab setup
tab_predict, tab_charts = st.tabs(["🔮 Real-Time Predictor", "📈 Model Dashboard & Charts"])

# -----------------------------------------------------------------------------
# TAB 1: REAL-TIME PREDICTOR
# -----------------------------------------------------------------------------
with tab_predict:
    col_upload, col_result = st.columns([1, 1], gap="large")
    
    with col_upload:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">📤 Upload Citra Benda Arkeologi</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Pilih file citra arkeologi (Format: JPG, JPEG, PNG, HEIC)",
            type=["jpg", "jpeg", "png", "heic"]
        )
        
        if uploaded_file is not None:
            try:
                # Open image with PIL
                pil_image = Image.open(uploaded_file).convert("RGB")
                st.image(pil_image, caption="Citra Asli yang Diupload", use_container_width=True)
                
                # Preprocess (Lanczos resize to 224x224, range [0, 255])
                img_tensor, img_resized = preprocess_for_prediction(pil_image)
            except Exception as e:
                st.error(f"Gagal memproses file gambar: {e}")
        else:
            st.info("💡 Unggah foto salah satu artefak museum di atas untuk memprediksi klaster dan kategorinya secara real-time.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display Preprocessing details side-by-side if image uploaded
        if uploaded_file is not None:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">⚙️ Image Resizing & Normalization Visualizer</div>', unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            c1.image(pil_image, caption="1. Citra Asli (Original)", use_container_width=True)
            c2.image(img_resized, caption="2. Model Input (224x224 Lanczos Resized)", use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">📊 Hasil Pengelompokan & Analisis Laten</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Batch expansion for prediction
            batch_img = np.expand_dims(img_tensor, axis=0) # Shape: (1, 224, 224, 3)
            
            # Predict soft assignment q and embedding z
            q_probs = dec_model.predict(batch_img, verbose=0)[0] # 10D soft assignments
            z_embed = enc_model.predict(batch_img, verbose=0)[0] # 10D normalized features
            
            predicted_cluster = np.argmax(q_probs)
            mapped_category = CLASS_NAMES[predicted_cluster]
            confidence = q_probs[predicted_cluster]
            
            # Polish display metrics
            st.markdown(f"""
            <div style="background: rgba(59, 130, 246, 0.15); border-radius: 12px; padding: 20px; border-left: 5px solid #3B82F6; margin-bottom: 20px;">
                <div style="font-size: 0.9rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Klaster Terprediksi</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #F8FAFC; margin: 5px 0;">Cluster {predicted_cluster:02d}</div>
                <div style="font-size: 1.15rem; font-weight: 600; color: #60A5FA;">🏛️ {mapped_category}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence slider
            st.write(f"**Confidence Score (Cluster Probability):** `{confidence*100:.2f}%`")
            st.progress(float(confidence))
            
            # 10D Softmax Bar Plot
            st.markdown("<br><b>Distribusi Assignment Softmax $q_{ij}$ (10-Dimensi):</b>", unsafe_allow_html=True)
            chart_data = {CLASS_NAMES[i]: float(q_probs[i]) for i in range(10)}
            st.bar_chart(chart_data)
            
            st.markdown("<br><b>Representasi Vektor Laten Ter-L2-Normalisasi ($z_i$):</b>", unsafe_allow_html=True)
            st.area_chart(z_embed)
            
        else:
            st.warning("⚠️ Silakan upload foto benda arkeologi di panel sebelah kiri untuk menampilkan hasil analisis klaster.")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display Gallery of other images in the same cluster!
        if uploaded_file is not None:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="card-header">🖼️ Citra Sejenis dalam Klaster {predicted_cluster:02d} ({mapped_category})</div>', unsafe_allow_html=True)
            
            # Search for sample files from the saved clusters
            cluster_folder_name = f"cluster_{predicted_cluster:02d}__{mapped_category.replace(' ', '_')}"
            target_cluster_path = os.path.join(CLUSTER_DIR, cluster_folder_name)
            
            # Search within CLUSTER_DIR (resolved relative to BASE_DIR)
            
            if os.path.exists(target_cluster_path):
                files = [os.path.join(target_cluster_path, f) for f in os.listdir(target_cluster_path) if f.lower().endswith(('.jpg','.jpeg','.png','.heic')) and not f.startswith('.')]
                
                if len(files) > 0:
                    st.write("Berikut adalah citra museum lain yang secara otomatis masuk ke dalam klaster sejenis:")
                    # Pick 4 random or sequential files to display
                    display_files = files[:min(4, len(files))]
                    cols = st.columns(len(display_files))
                    for idx, filepath in enumerate(display_files):
                        try:
                            # Convert HEIC to PIL if needed
                            gallery_img = Image.open(filepath)
                            cols[idx].image(gallery_img, use_container_width=True, caption=f"Sample {idx+1}")
                        except:
                            pass
                else:
                    st.write("Belum ada citra sampel dalam folder klaster ini.")
            else:
                st.write("Folder manifest klaster sejenis tidak dapat dimuat. Pastikan folder `hasil_dec_softmax_10d/cluster_dec/` berada pada direktori yang tepat.")
            
            st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: MODEL DASHBOARD & CHARTS
# -----------------------------------------------------------------------------
with tab_charts:
    col_g1, col_g2 = st.columns(2, gap="large")
    
    with col_g1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">📈 Progress Training & Silhouette Score</div>', unsafe_allow_html=True)
        
        train_chart_path = os.path.join(GRAFIK_DIR, "00_dec_training.png")
        if os.path.exists(train_chart_path):
            st.image(train_chart_path, caption="Progress Loss (KL-Divergence) & Silhouette Score Selama Refinement DEC", use_container_width=True)
        else:
            st.error("Grafik `00_dec_training.png` tidak ditemukan.")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">📊 Heatmap Confusion Matrix Evaluasi Klaster</div>', unsafe_allow_html=True)
        
        conf_chart_path = os.path.join(GRAFIK_DIR, "02_confusion.png")
        if os.path.exists(conf_chart_path):
            st.image(conf_chart_path, caption="Heatmap Akurasi Klasifikasi 1-to-1 Ground Truth vs DEC Cluster", use_container_width=True)
        else:
            st.error("Grafik `02_confusion.png` tidak ditemukan.")
            
        st.markdown('</div>', unsafe_allow_html=True)

    with col_g2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">🗺️ Visualisasi Sebaran Dimensi t-SNE & UMAP 2D</div>', unsafe_allow_html=True)
        
        tsne_chart_path = os.path.join(GRAFIK_DIR, "01_tsne_umap.png")
        if os.path.exists(tsne_chart_path):
            st.image(tsne_chart_path, caption="Peta 2D Proyeksi Ruang Laten 10D Menggunakan t-SNE dan UMAP", use_container_width=True)
        else:
            st.error("Grafik `01_tsne_umap.png` tidak ditemukan.")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">📉 Analisis Koefisien Siluet per Klaster</div>', unsafe_allow_html=True)
        
        sil_chart_path = os.path.join(GRAFIK_DIR, "03_silhouette.png")
        if os.path.exists(sil_chart_path):
            st.image(sil_chart_path, caption="Plot Koefisien Siluet Per-Sampel untuk Menunjukkan Kerapatan Klaster", use_container_width=True)
        else:
            st.error("Grafik `03_silhouette.png` tidak ditemukan.")
            
        st.markdown('</div>', unsafe_allow_html=True)
