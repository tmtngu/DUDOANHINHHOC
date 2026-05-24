import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN TRANG WEB
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
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    h1 { color: #1E8449; text-align: center; }
    .stRadio>div { gap: 20px; }
    </style>
""", unsafe_allow_html=True)

st.title("✨ HỆ THỐNG NHẬN DIỆN 16 HÌNH HỌC AI ✨")
st.divider()

# ==========================================
# 2. THANH CẤU HÌNH SIDEBAR (BỘ LỌC FIX LỖI VALUEERROR)
# ==========================================
st.sidebar.header("⚙️ BỘ CẤU HÌNH PHÙ HỢP MODEL")
st.sidebar.markdown("Điều chỉnh thông số dưới đây để sửa lỗi lệch cấu trúc với Colab:")

# Cấu hình kích thước ảnh
size_option = st.sidebar.selectbox(
    "1. Kích thước ảnh (Target Size):",
    ("224x224 (Mặc định Dataset)", "64x64"),
    index=0
)
TARGET_SIZE = 224 if "224x224" in size_option else 64

# Cấu hình số kênh màu -> CHÌA KHÓA FIX LỖI VALUEERROR KHỚP VỚI COLAB
channel_option = st.sidebar.selectbox(
    "2. Số kênh màu đầu vào (Channels):",
    ("3 kênh (RGB) - Mặc định trên Colab", "1 kênh (Grayscale / Xám)"),
    index=0
)
IS_RGB = "3 kênh" in channel_option

# Cấu hình Chuẩn hóa pixel
USE_RESCOLED = st.sidebar.checkbox(
    "3. Chuẩn hóa dữ liệu (Chia ảnh cho 255.0)", 
    value=True
)

# Cấu hình Auto-Crop
USE_AUTOCROP = st.sidebar.checkbox(
    "4. Cắt ảnh sát viền nét vẽ (Auto-Crop)", 
    value=False
)
st.sidebar.caption("💡 *Mẹo:* Nếu AI nhận diện sai lệch, hãy thử TẮT hoặc BẬT Auto-Crop để thử nghiệm.")

# Hiển thị danh sách 16 hình học hệ thống hỗ trợ
st.sidebar.divider()
st.sidebar.markdown("### 📊 Danh sách 16 hình học AI nhận diện:")
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
    st.error(f"⚠️ Lỗi không tìm thấy file mô hình 'model_nhandienhinhhoc_v3.keras': {e}")
    st.stop()


# ==========================================
# 4. GIAO DIỆN CHÍNH: KHU VỰC VẼ TAY KHÔNG THANH TRƯỢT
# ==========================================
col1, col2 = st.columns([5, 5], gap="large")

with col1:
    st.subheader("I. Khu vực vẽ tay & Chọn công cụ")
    
    # Cho phép chọn giữa vẽ tự do hoặc dùng tool kéo thả hình học chuẩn đét
    canvas_tool = st.radio(
        "Chọn chế độ thao tác trên bảng:", 
        ("🖌️ Vẽ tự do (Freedraw)", "⭕ Giữ chuột kéo Hình Tròn chuẩn", "🟦 Giữ chuột kéo Hình Vuông / Chữ nhật"), 
        horizontal=True
    )
    
    canvas_mode_map = {
        "🖌️ Vẽ tự do (Freedraw)": "freedraw", 
        "⭕ Giữ chuột kéo Hình Tròn chuẩn": "circle", 
        "🟦 Giữ chuột kéo Hình Vuông / Chữ nhật": "rect"
    )
    
    # Khởi tạo bảng vẽ canvas nền trắng nét đen sạch sẽ
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)", 
        stroke_width=4, 
        stroke_color="#000000",
        background_color="#FFFFFF", 
        height=350, 
        width=350,
        drawing_mode=canvas_mode_map[canvas_tool], 
        key="pure_drawing_canvas",
    )
    st.caption("💡 *Gợi ý:* Để vẽ hình tròn/hình vuông cân đối mượt mà nhất, ông nên tích chọn các công cụ kéo giữ ở trên!")

with col2:
    st.subheader("II. Kết quả dự đoán từ AI")
    
    if st.button("🚀 KÍCH HOẠT AI PHÂN TÍCH HÌNH VẼ", use_container_width=True):
        if canvas_result.image_data is not None:
            img_rgba = canvas_result.image_data
            
            # Kiểm tra xem người dùng đã đặt nét vẽ nào chưa
            if np.all(img_rgba[:, :, 3] == 0):
                st.warning("⚠️ Bảng vẽ đang trống! Ông hãy vẽ hoặc kéo thả một hình trước khi bấm phân tích nhé.")
            else:
                with st.spinner("🧠 AI đang phân tích dữ liệu hình học..."):
                    
                    # Chuyển đổi dữ liệu từ Canvas sang định dạng ảnh xám nền trắng nét đen
                    img_pil = Image.fromarray(img_rgba.astype('uint8'), 'RGBA')
                    white_bg = Image.new("RGB", img_pil.size, (255, 255, 255))
                    white_bg.paste(img_pil, mask=img_pil.split()[3])
                    gray_image = np.array(white_bg.convert("L"))
                    
                    # Thuật toán Auto-Crop nếu người dùng bật tính năng
                    if USE_AUTOCROP:
                        _, thresh = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY_INV)
                        coords = cv2.findNonZero(thresh)
                        if coords is not None:
                            x, y, w, h = cv2.boundingRect(coords)
                            pad = 15
                            x = max(0, x - pad); y = max(0, y - pad)
                            w = min(gray_image.shape[1] - x, w + 2*pad)
                            h = min(gray_image.shape[0] - y, h + 2*pad)
                            cropped = gray_image[y:y+h, x:x+w]
                            
                            max_side = max(w, h)
                            square_img = np.full((max_side, max_side), 255, dtype=np.uint8)
                            offset_x = (max_side - w) // 2; offset_y = (max_side - h) // 2
                            square_img[offset_y:offset_y+h, offset_x:offset_x+w] = cropped
                            gray_image = square_img

                    # Nhị phân hóa loại bỏ răng cưa và tiến hành resize chuẩn
                    _, binarized = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)
                    resized_image = cv2.resize(binarized, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
                    
                    # XỬ LÝ SỐ KÊNH MÀU ĐẦU VÀO ĐỂ TRIỆT TIÊU LỖI VALUEERROR
                    if IS_RGB:
                        # Convert từ 1 kênh xám sang 3 kênh màu RGB giống định dạng mặc định Colab
                        final_input = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
                        input_array = final_input.astype(np.float32)
                        if USE_RESCOLED:
                            input_array = input_array / 255.0
                        input_tensor = np.expand_dims(input_array, axis=0) # Kích thước tensor: (1, H, W, 3)
                    else:
                        # Giữ nguyên 1 kênh màu xám
                        input_array = resized_image.astype(np.float32)
                        if USE_RESCOLED:
                            input_array = input_array / 255.0
                        input_tensor = np.expand_dims(input_array, axis=[0, -1]) # Kích thước tensor: (1, H, W, 1)
                    
                    # Đưa vào model dự đoán
                    predictions = model.predict(input_tensor)[0]
                    top_1_idx = np.argmax(predictions)
                    top_1_score = predictions[top_1_idx] * 100
                    
                # --- HIỂN THỊ KẾT QUẢ RA MÀN HÌNH ---
                st.success(f"🎉 Kết quả dự đoán: **{CLASS_NAMES[top_1_idx]}**")
                st.metric(label="Độ tự tin của AI", value=f"{top_1_score:.2f}%")
                
                st.image(resized_image, caption=f"🔍 Ảnh thực tế ({TARGET_SIZE}x{TARGET_SIZE}) nạp vào não AI", width=140)
                
                if top_1_score >= 80.0:
                    st.balloons()
                
                st.markdown("#### 📊 Top 3 khả năng cao nhất:")
                top_3_indices = np.argsort(predictions)[-3:][::-1]
                for idx in top_3_indices:
                    score = predictions[idx] * 100
                    st.write(f"**{CLASS_NAMES[idx]}**: {score:.1f}%")
                    st.progress(int(score))
        else:
            st.warning("⚠️ Đã xảy ra lỗi khởi tạo vùng canvas.")