import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# 1. Cấu hình giao diện & CSS High-End
st.set_page_config(page_title="Shape AI Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .main-title { 
        text-align: center; font-weight: 800; font-size: 3.5rem;
        background: -webkit-linear-gradient(#00f2fe, #4facfe);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .status-card {
        background: #161b22; padding: 20px; border-radius: 12px;
        border: 1px solid #30363d; margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Load Model & Cấu hình Class
@st.cache_resource
def load_ai():
    return tf.keras.models.load_model('Model_7Hinh_VuaVan.keras')

try:
    model = load_ai()
    st.sidebar.success("✅ AI Engine: Online")
except:
    st.sidebar.error("❌ Thiếu file .keras trong thư mục!")

# Thứ tự folder khi ông train
CLASSES = ['binhhanh', 'chunhat', 'hinhtron', 'hinhthoi', 'lucgiac', 'ngoisao', 'tamgiac']

# 3. Giao diện chính
st.markdown('<h1 class="main-title">SHAPE RECOGNITION 2.0</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#8b949e;'>Hệ thống nhận diện hình học thông minh dựa trên CNN Custom</p>", unsafe_allow_html=True)

col_canvas, col_result = st.columns([1.5, 1])

with col_canvas:
    st.markdown("### 🎨 Bảng vẽ tay")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)", stroke_width=7,
        stroke_color="#000000", background_color="#FFFFFF",
        height=450, width=650, drawing_mode="freedraw", key="pro_canvas"
    )

with col_result:
    st.markdown("### 🧠 Kết quả phân tích")
    if canvas_result.image_data is not None:
        img = cv2.cvtColor(canvas_result.image_data, cv2.COLOR_RGBA2GRAY)
        
        # Nếu đã vẽ (có pixel đen)
        if np.any(img < 255):
            # Preprocessing X-Ray
            pts = np.argwhere(img < 255)
            y_min, x_min = pts[:,0].min(), pts[:,1].min()
            y_max, x_max = pts[:,0].max(), pts[:,1].max()
            cropped = img[y_min:y_max+1, x_min:x_max+1]
            
            # Tạo padding & resize
            h, w = cropped.shape
            side = max(h, w) + 40
            final_canvas = np.ones((side, side), dtype=np.uint8) * 255
            final_canvas[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w] = cropped
            
            # Prediction
            proc_img = cv2.resize(final_canvas, (224, 224))
            proc_img = cv2.cvtColor(proc_img, cv2.COLOR_GRAY2RGB)
            pred = model.predict(np.expand_dims(proc_img, axis=0), verbose=0)[0]
            
            # Hiển thị
            idx = np.argmax(pred)
            st.markdown(f"""
                <div class='status-card'>
                    <p style='color:#8b949e; margin:0;'>Dự đoán chính xác nhất:</p>
                    <h1 style='color:#00f2fe; margin:0;'>{CLASSES[idx].upper()}</h1>
                    <p style='color:#4facfe; margin:0;'>Độ tự tin: {pred[idx]*100:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            # X-Ray Sidebar
            st.sidebar.image(proc_img, caption="AI X-Ray Vision (224x224)", use_container_width=True)
            
            # Bar chart
            for i, p in enumerate(pred):
                st.write(f"{CLASSES[i].capitalize()}")
                st.progress(float(p))
        else:
            st.info("Hãy vẽ gì đó để AI bắt đầu làm việc!")