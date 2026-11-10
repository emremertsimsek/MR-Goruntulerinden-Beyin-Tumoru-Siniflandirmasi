import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import torch.nn.functional as F

# =========================================================
# UI KONFİGÜRASYONU VE STİL
# =========================================================
st.set_page_config(page_title="NeuroScan AI | Analysis", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stMetric { 
        background-color: white; 
        border-radius: 12px; 
        padding: 15px !important;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    h2 { color: #1e3a8a; font-family: 'Inter', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- SABİTLER ---
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
MODEL_PATH = "ResNet50_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# MODEL VE ANALİZ FONKSİYONLARI
# =========================================================
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        st.error(f"Model yüklenemedi: {e}")
    model.to(DEVICE).eval()
    return model

def generate_gradcam_pp(model, input_tensor):
    target_layer = model.layer4[-1] 
    activations, gradients = [], []
    def forward_hook(_, __, output): activations.append(output)
    def backward_hook(_, grad_input, grad_output): gradients.append(grad_output[0])
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)
    
    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    class_idx = torch.argmax(probs).item()
    confidence = probs[0, class_idx].item()
    model.zero_grad()
    output[0, class_idx].backward()
    
    acts, grads = activations[0].detach()[0], gradients[0].detach()[0]
    weights = torch.sum(torch.pow(grads, 2) / (2*torch.pow(grads, 2) + 
              torch.sum(acts * torch.pow(grads, 3), dim=(1,2), keepdim=True) + 1e-8) 
              * F.relu(grads), dim=(1,2))
    cam = torch.sum(weights[:, None, None] * acts, dim=0)
    cam = F.relu(cam).cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    
    h1.remove(); h2.remove()
    return cam, class_idx, confidence, probs.detach().cpu().numpy()[0]

# =========================================================
# ANA ARAYÜZ
# =========================================================
model = load_model()


st.markdown("<h2 style='text-align: center;'>MR Sınıflandırıcı</h2>", unsafe_allow_html=True)

with st.sidebar:
    st.header("📋 Veri Girişi")
    uploaded = st.file_uploader("MR Dosyası Yükleyin", type=["jpg","png","jpeg"])
    st.divider()
    st.header("⚙️ Görselleştirme")
    view_mode = st.radio("Analiz Modu", ["Renkli Isı Haritası", "Siyah-Beyaz Odak Maskesi"])
    opacity = st.slider("Şeffaflık", 0.1, 1.0, 0.45)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    img_np = np.array(img.resize((224,224)))

    cam, pred_idx, confidence, probs = generate_gradcam_pp(model, img_tensor)
    cam_resized = cv2.resize(cam, (224, 224))

    # --- ÜST METRİKLER ---
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Olası Sınıflandırma", CLASS_NAMES[pred_idx])
    with m2:
        st.metric("Güven Skoru", f"%{confidence*100:.2f}")
    with m3:
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        st.metric("Belirsizlik Değeri", f"{entropy:.3f}")

    st.divider()

    # --- GÖRSELLEŞTİRME KOLONLARI ---
    col_left, col_mid, col_right = st.columns([1, 1, 1])
    
    with col_left:
        st.markdown("**1. Giriş Görüntüsü**")
        st.image(img_np, use_container_width=True)
    
    with col_mid:
        st.markdown(f"**2. {view_mode}**")
        if view_mode == "Renkli Isı Haritası":
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            display_img = cv2.addWeighted(img_np, 1-opacity, heatmap, opacity, 0)
        else:
            mask = np.where(cam_resized > 0.4, 255, 0).astype(np.uint8)
            display_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        st.image(display_img, use_container_width=True)
    
    with col_right:
        st.markdown("**3. Olasılık Dağılımı**")
        chart_data = pd.DataFrame({"Olasılık": probs}, index=CLASS_NAMES)
        st.bar_chart(chart_data, color="#3b82f6")

    # --- AKADEMİK VERİ TABLOSU ---
    st.divider()
    st.subheader("📊 Analitik Veriler")
    
    y_max, x_max = np.unravel_index(np.argmax(cam_resized), cam_resized.shape)
    
    analysis_data = {
        "Parametre": [
            "Odak Merkezi Koordinatı (X, Y)", 
            "Maksimum Aktivasyon", 
            "Ortalama Yoğunluk", 
            "Efektif Odak Alanı (%)"
        ],
        "Değer": [
            f"{x_max}, {y_max}", 
            f"{np.max(cam_resized):.4f}", 
            f"{np.mean(cam_resized):.4f}", 
            f"{(np.sum(cam_resized > 0.5) / (224*224))*100:.2f}"
        ]
    }
    st.table(pd.DataFrame(analysis_data))

    # --- DURUM BİLDİRİMİ ---
    if CLASS_NAMES[pred_idx] == "No Tumor":
        st.success("Sistem Analizi: Patolojik işaretleyici saptanmadı.")
    else:
        st.warning(f"Sistem Notu: {CLASS_NAMES[pred_idx]} ile uyumlu alanlar gözlemlendi.")

else:
    st.info("Analizi başlatmak için bir görüntü yükleyiniz.")