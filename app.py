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

st.title("✨ HỆ THỐNG NHẬN DIỆN HÌNH HỌC AI ✨")
st.markdown("<p style='text-align: center; font-size:18px;'>Vẽ một hình học cơ bản vào bảng bên dưới, Trí tuệ nhân tạo sẽ phân tích nét vẽ của bạn!</p>", unsafe_allow_html=True)
st.divider()

# ==========================================
# 2. LOAD MODEL & DANH SÁCH NHÃN
# ==========================================
@st.cache_resource
def load_ai_model():
    # Load file mô hình vào bộ nhớ tạm để web chạy nhanh
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
col1, col2 = st.columns([4, 6], gap="large")

with col1:
    st.subheader("🖍️ Khu vực vẽ")
    st.caption("Hãy vẽ hình to, liền mạch và nằm gọn ở giữa khung trắng.")
    
    # Tạo bảng vẽ (Đã tăng stroke_width lên 8 để giữ nét khi thu nhỏ)
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)", # Nền trong suốt lúc vẽ
        stroke_width=8, # Nét vẽ đậm hơn để AI dễ nhìn
        stroke_color="#000000", # Nét vẽ màu đen
        background_color="#FFFFFF", # Nền bảng trắng
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
            img_rgba = canvas_result.image_data
            
            # Kiểm tra xem m có vẽ gì chưa (Kiểm tra kênh Alpha)
            if np.all(img_rgba[:, :, 3] == 0):
                st.warning("⚠️ Bảng vẽ đang trống, m nhớ vẽ hình vào trước khi bấm phân tích nha!")
            else:
                with st.spinner("🧠 AI đang căng não phân tích..."):
                    # --- BẮT ĐẦU QUÁ TRÌNH TIỀN XỬ LÝ ẢNH CHUẨN MỰC ---
                    
                    # 1. Chuyển ma trận Numpy sang dạng ảnh PIL
                    img_pil = Image.fromarray(img_rgba.astype('uint8'), 'RGBA')
                    
                    # 2. Tạo một lớp nền trắng bóc
                    white_bg = Image.new("RGB", img_pil.size, (255, 255, 255))
                    
                    # 3. Dán nét vẽ lên nền trắng (Sử dụng kênh Alpha làm mặt nạ) -> XÓA SẠCH LỖI NỀN ĐEN!
                    white_bg.paste(img_pil, mask=img_pil.split()[3])
                    
                    # 4. Chuyển thành ảnh xám (Grayscale)
                    gray_image = white_bg.convert("L")
                    
                    # 5. Ép lại thành Numpy Array và thu nhỏ về 64x64 (Dùng INTER_AREA giữ nét cực tốt)
                    gray_array = np.array(gray_image)
                    resized_image = cv2.resize(gray_array, (64, 64), interpolation=cv2.INTER_AREA)
                    
                    # 6. Chuẩn hóa Pixel về [0, 1] và Reshape
                    normalized_image = resized_image / 255.0
                    input_tensor = np.expand_dims(normalized_image, axis=[0, -1])

                    # --- AI BẮT ĐẦU ĐOÁN ---
                    predictions = model.predict(input_tensor)[0]
                    
                    top_1_idx = np.argmax(predictions)
                    top_1_score = predictions[top_1_idx] * 100
                    
                # Hiển thị vinh danh kết quả cao nhất
                st.success(f"🎉 Đây chắc chắn là: **{CLASS_NAMES[top_1_idx]}**")
                st.metric(label="Độ tự tin của AI", value=f"{top_1_score:.2f}%")
                
                # Hiệu ứng nổ bóng bay nếu AI tự tin trên 80%
                if top_1_score >= 80.0:
                    st.balloons()
                
                # Bonus: Vẽ thanh Progress Bar cho Top 3 dự đoán gần giống nhất
                st.markdown("#### 📊 Phân tích chuyên sâu (Top 3):")
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
    2. Cố gắng vẽ hình **to**, **cân đối** ở giữa khung.
    3. Vẽ xong bấm **Phân tích**.
    4. Bấm biểu tượng 🗑️ ở bảng vẽ để xóa và vẽ lại.
    """)
    st.divider()
    st.caption("🧠 Mô hình: Convolutional Neural Network (CNN) - 30 Epochs")
    st.caption("🔧 Hệ sinh thái: TensorFlow & Keras")
    st.caption("✨ Cập nhật: Khắc phục lỗi nền Alpha & Tối ưu nét vẽ")