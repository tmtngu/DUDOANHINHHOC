import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN TRANG WEB
# ==========================================
st.set_page_config(page_title="AI Nhận Diện Hình Học", page_icon="📐", layout="wide")

st.markdown("""
    <style>
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        transition: all 0.3s ease;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    h1 { color: #1E8449; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("✨ HỆ THỐNG NHẬN DIỆN HÌNH HỌC AI (V3) ✨")
st.markdown("<p style='text-align: center; font-size:18px;'>Tích hợp Auto-Crop & Khử Nhòe Nét Vẽ (Binarize)</p>", unsafe_allow_html=True)
st.divider()

# ==========================================
# 2. LOAD MODEL & DANH SÁCH NHÃN CHUẨN
# ==========================================
@st.cache_resource
def load_ai_model():
    # Đảm bảo file này nằm cùng thư mục với app.py
    return tf.keras.models.load_model("model_nhandienhinhhoc_v3.keras")

try:
    model = load_ai_model()
except Exception as e:
    st.error(f"⚠️ Lỗi không tìm thấy file mô hình: {e}\n\nÔng nhớ để file 'model_nhandienhinhhoc_v3.keras' chung thư mục với file app.py nhé!")
    st.stop()

# 16 Class khớp 100% với Colab
CLASS_NAMES = [
    "Hình bát giác (8 cạnh)",   # batgiac
    "Hình bình hành",           # binhhanh
    "Hình chữ nhật",            # chunhat
    "Hình cửu giác (9 cạnh)",   # cuugiac
    "Hình lục giác (6 cạnh)",   # lucgiac
    "Hình ngũ giác (5 cạnh)",   # ngugiac
    "Hình bán nguyệt",          # nuatron
    "Hình oval (Bầu dục)",      # oval
    "Hình ngôi sao",            # sao
    "Hình tam giác",            # tamgiac
    "Hình thang",               # thang
    "Hình thập giác (10 cạnh)", # thapgiac
    "Hình thất giác (7 cạnh)",  # thatgiac
    "Hình thoi",                # thoi
    "Hình tròn",                # tron
    "Hình vuông"                # vuong
]

# ==========================================
# 3. THIẾT KẾ LAYOUT CHÍNH
# ==========================================
col1, col2 = st.columns([4, 6], gap="large")

with col1:
    st.subheader("🖍️ Khu vực vẽ")
    st.caption("Hãy vẽ một hình duy nhất, rõ nét ở giữa khung.")
    
    # Set nét bút nhỏ (stroke_width=3) để giống với data train
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)", 
        stroke_width=2, 
        stroke_color="#000000", 
        background_color="#FFFFFF", 
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("🤖 Kết quả phân tích từ AI")
    st.caption("Bấm nút bên dưới sau khi vẽ xong.")
    
    if st.button("🚀 XÁC NHẬN VÀ PHÂN TÍCH", use_container_width=True):
        if canvas_result.image_data is not None:
            img_rgba = canvas_result.image_data
            
            if np.all(img_rgba[:, :, 3] == 0):
                st.warning("⚠️ Bảng vẽ đang trống, ông nhớ vẽ hình vào trước khi bấm phân tích nha!")
            else:
                with st.spinner("🧠 AI đang căng não phân tích..."):
                    # --- BƯỚC 1: XỬ LÝ ẢNH CƠ BẢN ---
                    img_pil = Image.fromarray(img_rgba.astype('uint8'), 'RGBA')
                    white_bg = Image.new("RGB", img_pil.size, (255, 255, 255))
                    white_bg.paste(img_pil, mask=img_pil.split()[3])
                    gray_image = np.array(white_bg.convert("L"))
                    
                    # --- BƯỚC 2: AUTO-CROP & SQUARE PADDING ---
                    _, thresh = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY_INV)
                    coords = cv2.findNonZero(thresh)
                    
                    if coords is not None:
                        x, y, w, h = cv2.boundingRect(coords)
                        pad = 10 
                        x = max(0, x - pad)
                        y = max(0, y - pad)
                        w = min(gray_image.shape[1] - x, w + 2*pad)
                        h = min(gray_image.shape[0] - y, h + 2*pad)
                        
                        cropped_image = gray_image[y:y+h, x:x+w]
                        
                        max_side = max(w, h)
                        square_img = np.full((max_side, max_side), 255, dtype=np.uint8)
                        offset_x = (max_side - w) // 2
                        offset_y = (max_side - h) // 2
                        square_img[offset_y:offset_y+h, offset_x:offset_x+w] = cropped_image
                        
                        resized_image = cv2.resize(square_img, (64, 64), interpolation=cv2.INTER_AREA)
                    else:
                        resized_image = cv2.resize(gray_image, (64, 64), interpolation=cv2.INTER_AREA)
                    
                    # --- BƯỚC 3: ÉP TRẮNG ĐEN (BINARIZE) KHỬ NHÒE ---
                    _, binarized_image = cv2.threshold(resized_image, 200, 255, cv2.THRESH_BINARY)

                    # --- BƯỚC 4: DỰ ĐOÁN (Không chia 255) ---
                    input_tensor = np.expand_dims(binarized_image, axis=[0, -1])
                    predictions = model.predict(input_tensor)[0]
                    
                    top_1_idx = np.argmax(predictions)
                    top_1_score = predictions[top_1_idx] * 100
                    
                # ==========================================
                # IN KẾT QUẢ RA MÀN HÌNH
                # ==========================================
                st.success(f"🎉 Đây chắc chắn là: **{CLASS_NAMES[top_1_idx]}**")
                st.metric(label="Độ tự tin của AI", value=f"{top_1_score:.2f}%")
                
                # Hiển thị camera ẩn để check nét vẽ
                st.image(binarized_image, caption="🔍 Ảnh thực tế (64x64) mà AI nhìn thấy", width=120)
                
                if top_1_score >= 80.0:
                    st.balloons()
                
                st.markdown("#### 📊 Phân tích chuyên sâu (Top 3 khả năng):")
                top_3_indices = np.argsort(predictions)[-3:][::-1]
                
                for idx in top_3_indices:
                    score = predictions[idx] * 100
                    st.write(f"**{CLASS_NAMES[idx]}**: {score:.1f}%")
                    st.progress(int(score))
        else:
            st.info("Vui lòng vẽ một hình để bắt đầu!")

# ==========================================
# 4. SIDEBAR - HƯỚNG DẪN
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2083/2083213.png", width=100)
    st.header("💡 Hướng dẫn sử dụng")
    st.markdown("""
    1. Dùng chuột vẽ hình vào bảng trắng bên trái.
    2. Vẽ xong bấm **Phân tích**.
    3. Hệ thống sẽ tự động căn chỉnh và khử viền xám.
    4. Bấm biểu tượng 🗑️ ở bảng vẽ để xóa và vẽ lại.
    """)
    st.divider()
    st.caption("🧠 Mô hình: CNN Deep Learning V3")
    st.caption("🔧 Tích hợp Binarize khử nhòe nét vẽ")