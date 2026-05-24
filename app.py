import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN TRANG WEB
# ==========================================
st.set_page_config(page_title="AI Nhận Diện Hình Học", page_icon="📐", layout="wide")

# Thêm CSS để làm nút bấm và layout lấp lánh hơn
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
    }
    .stButton>button:hover {
        background-color: #ff3333;
        transform: scale(1.02);
    }
    h1 { color: #1E8449; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("✨ HỆ THỐNG NHẬN DIỆN HÌNH HỌC AI ✨")
st.markdown("<p style='text-align: center; font-size:18px;'>Vẽ một hình học cơ bản vào bảng bên dưới, Trí tuệ nhân tạo sẽ phân tích nét vẽ của bạn!</p>", unsafe_allow_html=True)
st.divider()

# ==========================================
# 2. LOAD MODEL & DANH SÁCH NHÃN
# ==========================================
@st.cache_resource
def load_ai_model():
    # Load file 30MB của m vào bộ nhớ tạm để web chạy nhanh
    return tf.keras.models.load_model("mo_hinh_chuan_version.keras")

model = load_ai_model()

CLASS_NAMES = [
    "Hình bát giác (8 cạnh)", "Hình bình hành", "Hình chữ nhật",
    "Hình cửu giác (9 cạnh)", "Hình lục giác (6 cạnh)", "Hình ngũ giác (5 cạnh)",
    "Hình bán nguyệt (Nửa tròn)", "Hình oval (Bầu dục)", "Hình ngôi sao",
    "Hình tam giác", "Hình thang", "Hình thập giác (10 cạnh)",
    "Hình thất giác (7 cạnh)", "Hình thoi", "Hình tròn", "Hình vuông"
]

# ==========================================
# 3. THIẾT KẾ LAYOUT CHIA CỘT (CANVAS & KẾT QUẢ)
# ==========================================
# Cột trái (Bảng vẽ) chiếm 40%, Cột phải (Kết quả) chiếm 60%
col1, col2 = st.columns([4, 6], gap="large")

with col1:
    st.subheader("🖍️ Khu vực vẽ")
    st.caption("Hãy vẽ hình bằng một nét rõ ràng, liền mạch ở giữa khung.")
    
    # Tạo bảng vẽ xịn
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.3)",
        stroke_width=4,
        stroke_color="#000000", # Nét vẽ màu đen
        background_color="#FFFFFF", # Nền trắng giống giấy
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("🤖 Kết quả phân tích từ AI")
    st.caption("Bấm nút bên dưới sau khi vẽ xong để hệ thống quét ảnh.")
    
    if st.button("🚀 XÁC NHẬN VÀ PHÂN TÍCH", use_container_width=True):
        if canvas_result.image_data is not None:
            # 1. Trích xuất ma trận điểm ảnh từ bảng vẽ (Định dạng RGBA)
            img_rgba = canvas_result.image_data
            
            # Kiểm tra xem m có vẽ gì chưa (Nếu toàn màu trắng nền thì báo lỗi)
            if np.all(img_rgba[:, :, 3] == 0):
                st.warning("⚠️ Bảng vẽ đang trống, m nhớ vẽ hình vào trước khi bấm phân tích nha!")
            else:
                # 2. Xử lý ảnh (Khớp 100% với cách AI đã học)
                # Chuyển hệ màu RGBA sang Grayscale (Ảnh xám)
                gray_image = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2GRAY)
                # Thu nhỏ ảnh về đúng kích thước 64x64
                resized_image = cv2.resize(gray_image, (64, 64))
                # Chuẩn hóa giá trị Pixel từ [0,255] về [0,1]
                normalized_image = resized_image / 255.0
                # Thêm chiều (batch_size) để đưa vào mạng CNN: (1, 64, 64, 1)
                input_tensor = np.expand_dims(normalized_image, axis=[0, -1])

                # 3. AI bắt đầu đoán
                predictions = model.predict(input_tensor)[0]
                
                # Tìm ra nhãn có xác suất cao nhất
                top_1_idx = np.argmax(predictions)
                top_1_score = predictions[top_1_idx] * 100
                
                # Hiển thị vinh danh kết quả cao nhất
                st.success(f"🎉 Đây chắc chắn là: **{CLASS_NAMES[top_1_idx]}**")
                st.metric(label="Độ tự tin của AI", value=f"{top_1_score:.2f}%")
                
                # Bonus: Vẽ thanh Progress Bar cho Top 3 dự đoán gần giống nhất
                st.markdown("#### 📊 Phân tích chuyên sâu (Top 3):")
                top_3_indices = np.argsort(predictions)[-3:][::-1]
                
                for idx in top_3_indices:
                    score = predictions[idx] * 100
                    st.write(f"**{CLASS_NAMES[idx]}**: {score:.1f}%")
                    st.progress(int(score))
        else:
            st.info("Vui lòng vẽ một hình để bắt đầu!")

# Thanh bên (Sidebar) giải thích thông tin
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2083/2083213.png", width=100)
    st.header("💡 Hướng dẫn sử dụng")
    st.markdown("""
    1. Dùng chuột vẽ hình vào bảng trắng bên trái.
    2. Cố gắng vẽ hình to, cân đối ở giữa khung.
    3. Vẽ xong bấm **Phân tích**.
    4. Bấm biểu tượng 🗑️ ở bảng vẽ để xóa và vẽ lại.
    """)
    st.divider()
    st.caption("🧠 Mô hình: Convolutional Neural Network (CNN) - 30 Epochs")
    st.caption("🔧 Hệ sinh thái: TensorFlow 2.21 & Keras 3.12")