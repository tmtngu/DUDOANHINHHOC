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
st.markdown("<p style='text-align: center; font-size:18px;'>Tích hợp Auto-Crop Bounding Box & Square Padding chống méo hình</p>", unsafe_allow_html=True)
st.divider()

# ==========================================
# 2. LOAD MODEL & DANH SÁCH NHÃN (KHỚP 100% VỚI COLAB)
# ==========================================
@st.cache_resource
def load_ai_model():
    # Load phiên bản V3 xịn nhất vừa train
    return tf.keras.models.load_model("model_nhandienhinhhoc_v3.keras")

try:
    model = load_ai_model()
except Exception as e:
    st.error(f"⚠️ Lỗi không tìm thấy file mô hình: {e}\n\nÔng nhớ để file 'model_nhandienhinhhoc_v3.keras' chung thư mục với file app.py nhé!")
    st.stop()

# Danh sách đã được xếp đúng thứ tự Alphabet giống hệt trên thư mục train
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
# 3. THIẾT KẾ LAYOUT
# ==========================================
col1, col2 = st.columns([4, 6], gap="large")

with col1:
    st.subheader("🖍️ Khu vực vẽ")
    st.caption("Cứ vẽ thoải mái ở giữa khung, AI sẽ tự động cắt lề và căn chỉnh!")
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)", 
        stroke_width=1, 
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
            
            # Kiểm tra xem bảng vẽ có trống không
            if np.all(img_rgba[:, :, 3] == 0):
                st.warning("⚠️ Bảng vẽ đang trống, ông nhớ vẽ hình vào trước khi bấm phân tích nha!")
            else:
                with st.spinner("🧠 AI đang căng não phân tích..."):
                    # 1. Chuyển RGBA sang PIL và đổ nền trắng
                    img_pil = Image.fromarray(img_rgba.astype('uint8'), 'RGBA')
                    white_bg = Image.new("RGB", img_pil.size, (255, 255, 255))
                    white_bg.paste(img_pil, mask=img_pil.split()[3])
                    
                    # 2. Chuyển thành Numpy Array dạng Grayscale
                    gray_image = np.array(white_bg.convert("L"))
                    
                    # --- AUTO-CROP BẢO TOÀN TỶ LỆ ---
                    # Tìm tọa độ nét vẽ bằng cách đảo màu (đen thành trắng)
                    _, thresh = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY_INV)
                    coords = cv2.findNonZero(thresh)
                    
                    if coords is not None:
                        # Vẽ Bounding Box
                        x, y, w, h = cv2.boundingRect(coords)
                        
                        # Thêm padding nhẹ 15 pixel để viền không chạm sát vách
                        pad = 15
                        x = max(0, x - pad)
                        y = max(0, y - pad)
                        w = min(gray_image.shape[1] - x, w + 2*pad)
                        h = min(gray_image.shape[0] - y, h + 2*pad)
                        
                        # Cắt hình
                        cropped_image = gray_image[y:y+h, x:x+w]
                        
                        # Square Padding: Tạo khung vuông màu trắng bằng với cạnh dài nhất
                        max_side = max(w, h)
                        square_img = np.full((max_side, max_side), 255, dtype=np.uint8)
                        
                        # Dán hình vừa cắt vào chính giữa khung vuông
                        offset_x = (max_side - w) // 2
                        offset_y = (max_side - h) // 2
                        square_img[offset_y:offset_y+h, offset_x:offset_x+w] = cropped_image
                        
                        # Resize khung vuông về 64x64
                        resized_image = cv2.resize(square_img, (64, 64), interpolation=cv2.INTER_AREA)
                    else:
                        resized_image = cv2.resize(gray_image, (64, 64), interpolation=cv2.INTER_AREA)
                    
                    # KHÔNG CHIA 255 (Vì Model V3 đã có layers.Rescaling)
                    input_tensor = np.expand_dims(resized_image, axis=[0, -1])

                    # Dự đoán
                    predictions = model.predict(input_tensor)[0]
                    
                    top_1_idx = np.argmax(predictions)
                    top_1_score = predictions[top_1_idx] * 100
                    
                # Hiển thị kết quả
                st.success(f"🎉 Đây chắc chắn là: **{CLASS_NAMES[top_1_idx]}**")
                st.metric(label="Độ tự tin của AI", value=f"{top_1_score:.2f}%")
                
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
# 4. SIDEBAR - THANH THÔNG TIN BÊN LỀ
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2083/2083213.png", width=100)
    st.header("💡 Hướng dẫn sử dụng")
    st.markdown("""
    1. Dùng chuột vẽ hình vào bảng trắng bên trái.
    2. Vẽ xong bấm **Phân tích**.
    3. Hệ thống sẽ tự động căn chỉnh, cắt lề tỷ lệ vàng.
    4. Bấm biểu tượng 🗑️ ở bảng vẽ để xóa và vẽ lại.
    """)
    st.divider()
    st.caption("🧠 Mô hình: CNN Deep Learning V3")
    st.caption("🔧 Tích hợp Auto-Crop & Square Padding")