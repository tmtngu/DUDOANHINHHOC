import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN ỨNG DỤNG
# ==========================================
st.set_page_config(page_title="AI Nhận Diện 16 Hình Học", page_icon="📐", layout="wide")

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
    }
    h1 { color: #1E8449; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("✨ HỆ THỐNG NHẬN DIỆN 16 HÌNH HỌC AI ✨")
st.divider()

# ==========================================
# 2. THANH CẤU HÌNH SIDEBAR (FIX LỖI VALUEERROR)
# ==========================================
st.sidebar.header("⚙️ BỘ CẤU HÌNH PHÙ HỢP MODEL")
st.sidebar.markdown("Điều chỉnh cấu hình để trùng khớp hoàn toàn với lúc train model trên Colab:")

size_option = st.sidebar.selectbox(
    "1. Kích thước ảnh (Target Size):",
    ("224x224 (Mặc định Dataset)", "64x64"),
    index=0
)
TARGET_SIZE = 224 if "224x224" in size_option else 64

channel_option = st.sidebar.selectbox(
    "2. Số kênh màu (Channels):",
    ("3 kênh (RGB) - Mặc định trên Colab", "1 kênh (Grayscale)"),
    index=0
)
IS_RGB = "3 kênh" in channel_option

USE_RESCOLED = st.sidebar.checkbox("3. Chuẩn hóa dữ liệu (Chia cho 255.0)", value=True)
USE_AUTOCROP = st.sidebar.checkbox("4. Cắt ảnh sát viền nét vẽ (Auto-Crop)", value=False)

st.sidebar.divider()
st.sidebar.markdown("### 📊 Danh sách 16 hình học của hệ thống:")
CLASS_NAMES = [
    "Hình bát giác (8 cạnh)", "Hình bình hành", "Hình chữ nhật", "Hình cửu giác (9 cạnh)",
    "Hình lục giác (6 cạnh)", "Hình ngũ giác (5 cạnh)", "Hình bán nguyệt", "Hình oval (Bầu dục)",
    "Hình ngôi sao", "Hình tam giác", "Hình thang", "Hình thập giác (10 cạnh)",
    "Hình thất giác (7 cạnh)", "Hình thoi", "Hình tròn", "Hình vuông"
]
for shape in CLASS_NAMES:
    st.sidebar.write(f"- {shape}")


# ==========================================
# 3. TẢI MÔ HÌNH AI (MODEL LOAD)
# ==========================================
@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model("model_nhandienhinhhoc_v3.keras")

try:
    model = load_ai_model()
except Exception as e:
    st.error(f"⚠️ Không tìm thấy file mô hình 'model_nhandienhinhhoc_v3.keras'. Lỗi: {e}")
    st.stop()


# ==========================================
# 4. GIAO DIỆN CHÍNH: 17 TÙY CHỌN VẼ
# ==========================================
col1, col2 = st.columns([5, 5], gap="large")

with col1:
    st.subheader("I. Khu vực bảng vẽ")
    
    # Đủ 17 lựa chọn theo ý ông
    options = [
        "🖌️ Vẽ tay tự do", 
        "⭕ Hình tròn", 
        "🟦 Hình vuông", 
        "▭ Hình chữ nhật", 
        "🔺 Hình tam giác", 
        "⭐ Hình ngôi sao", 
        "🔷 Hình thoi", 
        "⏢ Hình thang", 
        "⬟ Hình ngũ giác (5 cạnh)", 
        "⬡ Hình lục giác (6 cạnh)", 
        "⬢ Hình thất giác (7 cạnh)", 
        "🛑 Hình bát giác (8 cạnh)", 
        "💠 Hình cửu giác (9 cạnh)", 
        "💠 Hình thập giác (10 cạnh)", 
        "▱ Hình bình hành", 
        "🌙 Hình bán nguyệt", 
        "🥚 Hình oval (Bầu dục)"
    ]
    
    selected_option = st.selectbox("🎯 Chọn mục muốn vẽ:", options, index=0)
    
    # Logic mapping: Gắn từng mục với công cụ Canvas tương ứng
    if selected_option in ["🖌️ Vẽ tay tự do", "🌙 Hình bán nguyệt"]:
        drawing_mode = "freedraw"
        st.info("💡 **Hướng dẫn:** Nhấn giữ chuột trái và di chuyển để vẽ tự do.")
    elif selected_option in ["⭕ Hình tròn", "🥚 Hình oval (Bầu dục)"]:
        drawing_mode = "circle"
        st.info("💡 **Hướng dẫn:** Nhấn giữ chuột trái và kéo rộng ra để tạo hình.")
    elif selected_option in ["🟦 Hình vuông", "▭ Hình chữ nhật"]:
        drawing_mode = "rect"
        st.info("💡 **Hướng dẫn:** Nhấn giữ chuột trái và kéo rộng ra để tạo khối vuông/chữ nhật.")
    else:
        drawing_mode = "polygon"
        st.info(f"💡 **Hướng dẫn vẽ {selected_option.split(' ')[1]}:** Hãy click chuột từng điểm một trên bảng trắng để tạo các góc. Khi click xong điểm cuối cùng, hãy **Nhấp đúp chuột (Double-click)** để tự động khép kín hình nhé!")
    
    # Khởi tạo bảng vẽ canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=4,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=380,
        width=380,
        drawing_mode=drawing_mode,
        key="geometry_17options_canvas"
    )

with col2:
    st.subheader("II. Kết quả phân tích từ AI")
    
    if st.button("🚀 KÍCH HOẠT AI PHÂN TÍCH HÌNH VẼ", use_container_width=True):
        if canvas_result.image_data is not None:
            img_rgba = canvas_result.image_data
            
            if np.all(img_rgba[:, :, 3] == 0):
                st.warning("⚠️ Bảng vẽ đang trống rỗng! Ông hãy vẽ một hình lên bảng trước nhé.")
            else:
                with st.spinner("🧠 AI đang phân tích hình học..."):
                    img_pil = Image.fromarray(img_rgba.astype('uint8'), 'RGBA')
                    white_bg = Image.new("RGB", img_pil.size, (255, 255, 255))
                    white_bg.paste(img_pil, mask=img_pil.split()[3])
                    gray_image = np.array(white_bg.convert("L"))
                    
                    if USE_AUTOCROP:
                        _, thresh = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY_INV)
                        coords = cv2.findNonZero(thresh)
                        if coords is not None:
                            x, y, w, h = cv2.boundingRect(coords)
                            pad = 15
                            x = max(0, x - pad)
                            y = max(0, y - pad)
                            w = min(gray_image.shape[1] - x, w + 2*pad)
                            h = min(gray_image.shape[0] - y, h + 2*pad)
                            cropped = gray_image[y:y+h, x:x+w]
                            
                            max_side = max(w, h)
                            square_img = np.full((max_side, max_side), 255, dtype=np.uint8)
                            offset_x = (max_side - w) // 2
                            offset_y = (max_side - h) // 2
                            square_img[offset_y:offset_y+h, offset_x:offset_x+w] = cropped
                            gray_image = square_img

                    _, binarized = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)
                    resized_image = cv2.resize(binarized, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
                    
                    if IS_RGB:
                        final_input = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
                        input_array = final_input.astype(np.float32)
                        if USE_RESCOLED:
                            input_array = input_array / 255.0
                        input_tensor = np.expand_dims(input_array, axis=0)
                    else:
                        input_array = resized_image.astype(np.float32)
                        if USE_RESCOLED:
                            input_array = input_array / 255.0
                        input_tensor = np.expand_dims(input_array, axis=[0, -1])
                    
                    predictions = model.predict(input_tensor)[0]
                    top_1_idx = np.argmax(predictions)
                    top_1_score = predictions[top_1_idx] * 100
                    
                st.success(f"🎉 Kết quả dự đoán chuẩn nhất: **{CLASS_NAMES[top_1_idx]}**")
                st.metric(label="Độ chính xác / Tự tin", value=f"{top_1_score:.2f}%")
                
                st.image(resized_image, caption=f"🔍 Kích thước ảnh thực tế nạp vào AI", width=140)
                
                if top_1_score >= 85.0:
                    st.balloons()
                
                st.markdown("#### 📊 Top 3 khả năng cao nhất:")
                top_3_indices = np.argsort(predictions)[-3:][::-1]
                for idx in top_3_indices:
                    score = predictions[idx] * 100
                    st.write(f"**{CLASS_NAMES[idx]}**: {score:.1f}%")
                    st.progress(int(score))
        else:
            st.warning("⚠️ Vùng canvas gặp sự cố khởi tạo dữ liệu ảnh.")