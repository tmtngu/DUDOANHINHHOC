import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN ỨNG DỤNG
# ==========================================
st.set_page_config(page_title="Siêu AI Nhận Diện 16 Hình Học", page_icon="📐", layout="wide")

st.markdown("""
    <style>
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        transition: all 0.3s ease;
        border: none;
    }
    .stButton>button:hover {
        background-color: #27ae60;
        transform: scale(1.02);
    }
    h1 { color: #1E8449; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("✨ HỆ THỐNG NHẬN DIỆN 16 HÌNH HỌC (MÔ HÌNH TRÊN MÁY) ✨")
st.divider()

# ==========================================
# 2. CẤU HÌNH CỐ ĐỊNH THEO MODEL MOBILENETV2
# ==========================================
# Mô hình mới dùng MobileNetV2 bắt buộc kích thước là 224x224 và ảnh 3 kênh màu RGB
TARGET_SIZE = 224 

# 🛑 LƯU Ý QUAN TRỌNG: Thứ tự các hình dưới đây phải trùng khớp 100% với thứ tự 
# tên các thư mục con lúc ông train trên Colab (Keras tự sắp xếp theo bảng chữ cái ABC)
CLASS_NAMES = [
    "Hình bán nguyệt", "Hình bát giác (8 cạnh)", "Hình bình hành", "Hình chữ nhật", 
    "Hình cửu giác (9 cạnh)", "Hình lục giác (6 cạnh)", "Hình ngũ giác (5 cạnh)", "Hình ngôi sao", 
    "Hình oval (Bầu dục)", "Hình tam giác", "Hình thang", "Hình thập giác (10 cạnh)",
    "Hình thất giác (7 cạnh)", "Hình thoi", "Hình tròn", "Hình vuông"
]

# Thêm tùy chọn Auto-Crop ở thanh Sidebar nếu nét vẽ bị lệch tâm
st.sidebar.header("⚙️ CẤU HÌNH NÂNG CAO")
USE_AUTOCROP = st.sidebar.checkbox("Cắt ảnh sát viền nét vẽ (Auto-Crop)", value=False)

st.sidebar.divider()
st.sidebar.markdown("### 📊 Thứ tự 16 lớp hình học:")
for idx, shape in enumerate(CLASS_NAMES):
    st.sidebar.write(f"{idx}. {shape}")


# ==========================================
# 3. TẢI MÔ HÌNH AI MỚI (MODEL LOAD)
# ==========================================
@st.cache_resource
def load_smart_model():
    # Tải file .keras thông minh ông vừa train xong từ Colab về
    return tf.keras.models.load_model("model_nhandienhinhhoc_v3.keras")

try:
    model = load_smart_model()
except Exception as e:
    st.error(f"⚠️ Không tìm thấy file mô hình 'model_nhandienhinhhoc_v3.keras' trong thư mục code. Hãy download từ Colab về và vứt cạnh file app.py nhé! Lỗi chi tiết: {e}")
    st.stop()


# ==========================================
# 4. GIAO DIỆN CHÍNH: BẢNG VẼ VỚI 17 TÙY CHỌN
# ==========================================
col1, col2 = st.columns([5, 5], gap="large")

with col1:
    st.subheader("I. Khu vực bảng vẽ")
    
    options = [
        "🖌️ Vẽ tay tự do", "⭕ Hình tròn", "🟦 Hình vuông", "▭ Hình chữ nhật", 
        "🔺 Hình tam giác", "⭐ Hình ngôi sao", "🔷 Hình thoi", "⏢ Hình thang", 
        "⬟ Hình ngũ giác (5 cạnh)", "⬡ Hình lục giác (6 cạnh)", "⬢ Hình thất giác (7 cạnh)", 
        "🛑 Hình bát giác (8 cạnh)", "💠 Hình cửu giác (9 cạnh)", "💠 Hình thập giác (10 cạnh)", 
        "▱ Hình bình hành", "🌙 Hình bán nguyệt", "🥚 Hình oval (Bầu dục)"
    ]
    
    selected_option = st.selectbox("🎯 Chọn mục muốn vẽ:", options, index=0)
    
    # Bộ điều hướng công cụ tự động
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
        st.info(f"💡 **Hướng dẫn vẽ {selected_option.split(' ')[1]}:** Hãy click chuột từng điểm một trên bảng trắng để tạo các góc thẳng tắp. Khi vẽ xong góc cuối, hãy **Nhấp đúp chuột (Double-click)** để khép kín hình.")
    
    # Khởi tạo bảng vẽ Canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=4,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=380,
        width=380,
        drawing_mode=drawing_mode,
        key="smart_local_canvas"
    )

with col2:
    st.subheader("II. Kết quả phân tích từ AI Mới")
    
    if st.button("🚀 KÍCH HOẠT AI PHÂN TÍCH HÌNH VẼ", use_container_width=True):
        if canvas_result.image_data is not None:
            img_rgba = canvas_result.image_data
            
            if np.all(img_rgba[:, :, 3] == 0):
                st.warning("⚠️ Bảng vẽ đang trống! Ông hãy vẽ một hình lên trước nhé.")
            else:
                with st.spinner("🧠 Con AI thông minh đang phân tích nét vẽ..."):
                    # 1. Chuyển đổi dữ liệu sang nền trắng nét đen giống hệt Dataset gốc
                    img_pil = Image.fromarray(img_rgba.astype('uint8'), 'RGBA')
                    white_bg = Image.new("RGB", img_pil.size, (255, 255, 255))
                    white_bg.paste(img_pil, mask=img_pil.split()[3])
                    gray_image = np.array(white_bg.convert("L"))
                    
                    # Thuật toán Auto-Crop nếu bật
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

                    # 2. Làm mịn nét vẽ và resize về 224x224 chuẩn MobileNetV2
                    _, binarized = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)
                    resized_gray = cv2.resize(binarized, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
                    
                    # Chuyển từ ảnh xám sang ảnh 3 kênh màu RGB đúng định dạng mô hình yêu cầu
                    final_rgb = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2RGB)
                    
                    # Chuyển thành mảng float32 dữ liệu thô (0.0 -> 255.0)
                    # 🛑 KHÔNG chia cho 255.0 ở đây vì tầng Lambda bên trong mô hình mới đã tự xử lý rồi!
                    input_array = final_rgb.astype(np.float32)
                    input_tensor = np.expand_dims(input_array, axis=0)
                    
                    # 3. Cho AI dự đoán
                    predictions = model.predict(input_tensor)[0]
                    top_1_idx = np.argmax(predictions)
                    top_1_score = predictions[top_1_idx] * 100
                    
                # --- HIỂN THỊ KẾT QUẢ XỊN SÒ ---
                st.success(f"🎉 Kết quả dự đoán chuẩn nhất: **{CLASS_NAMES[top_1_idx]}**")
                st.metric(label="Độ chính xác / Tự tin", value=f"{top_1_score:.2f}%")
                
                st.image(resized_gray, caption="🔍 Ảnh 224x224 thực tế đang nạp vào bộ não AI", width=140)
                
                if top_1_score >= 80.0:
                    st.balloons()
                
                st.markdown("#### 📊 Top 3 khả năng cao nhất:")
                top_3_indices = np.argsort(predictions)[-3:][::-1]
                for idx in top_3_indices:
                    score = predictions[idx] * 100
                    st.write(f"**{CLASS_NAMES[idx]}**: {score:.1f}%")
                    st.progress(int(score))
        else:
            st.warning("⚠️ Vùng canvas gặp sự cố khởi tạo dữ liệu ảnh.")