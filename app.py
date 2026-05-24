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

st.title("✨ HỆ THỐNG NHẬN DIỆN HÌNH HỌC AI ✨")
st.markdown("<p style='text-align: center; font-size:18px;'>Hệ thống Auto-Crop: Tự động loại bỏ lề thừa, khớp 100% với dữ liệu Train!</p>", unsafe_allow_html=True)
st.divider()

# ==========================================
# 2. LOAD MODEL & DANH SÁCH NHÃN
# ==========================================
@st.cache_resource
def load_ai_model():
    # NHỚ ĐỔI TÊN FILE SANG BẢN V2 VỪA TRAIN XONG NHÉ
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
# 3. THIẾT KẾ LAYOUT CHIA CỘT
# ==========================================
col1, col2 = st.columns([4, 6], gap="large")

with col1:
    st.subheader("🖍️ Khu vực vẽ")
    st.caption("Hãy vẽ một hình bất kỳ. Hệ thống sẽ tự động bắt nét và cắt lề.")
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)", 
        stroke_width=6, # Giảm nét vẽ xuống xíu để AI dễ nhìn viền sắc cạnh
        stroke_color="#000000", 
        background_color="#FFFFFF", 
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
            
            if np.all(img_rgba[:, :, 3] == 0):
                st.warning("⚠️ Bảng vẽ đang trống, m nhớ vẽ hình vào trước khi bấm phân tích nha!")
            else:
                with st.spinner("🧠 AI đang căng não phân tích..."):
                    # 1. Chuyển RGBA sang PIL và đổ nền trắng
                    img_pil = Image.fromarray(img_rgba.astype('uint8'), 'RGBA')
                    white_bg = Image.new("RGB", img_pil.size, (255, 255, 255))
                    white_bg.paste(img_pil, mask=img_pil.split()[3])
                    
                    # 2. Chuyển thành Numpy Array dạng Grayscale
                    gray_image = np.array(white_bg.convert("L"))
                    
                    # --- BẮT ĐẦU THUẬT TOÁN AUTO-CROP ---
                    # 3. Đảo ngược màu (Trắng thành đen, đen nét vẽ thành trắng) để tìm tọa độ nét vẽ
                    _, thresh = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY_INV)
                    
                    # 4. Tìm tọa độ của tất cả các pixel nét vẽ
                    coords = cv2.findNonZero(thresh)
                    
                    if coords is not None:
                        # 5. Vẽ một hình chữ nhật bao quanh khít nhất nét vẽ (Bounding Box)
                        x, y, w, h = cv2.boundingRect(coords)
                        
                        # Thêm tí lề (padding) 10 pixel xung quanh để hình không bị cắt sát rạt
                        pad = 10
                        x = max(0, x - pad)
                        y = max(0, y - pad)
                        w = min(gray_image.shape[1] - x, w + 2*pad)
                        h = min(gray_image.shape[0] - y, h + 2*pad)
                        
                        # 6. CẮT ẢNH: Vứt bỏ phần lề trắng bao la bên ngoài
                        cropped_image = gray_image[y:y+h, x:x+w]
                        
                        # 7. Thu nhỏ cái ảnh đã cắt về đúng 64x64
                        resized_image = cv2.resize(cropped_image, (64, 64), interpolation=cv2.INTER_AREA)
                    else:
                        # Backup lỡ thuật toán không tìm thấy viền thì xài ảnh gốc
                        resized_image = cv2.resize(gray_image, (64, 64), interpolation=cv2.INTER_AREA)
                    # --- KẾT THÚC AUTO-CROP ---

                    # 8. Định dạng lại Tensor để truyền vào model (KHÔNG chia 255 nữa vì model V2 đã tự lo)
                    input_tensor = np.expand_dims(resized_image, axis=[0, -1])

                    # 9. Đẩy vào Model dự đoán
                    predictions = model.predict(input_tensor)[0]
                    
                    top_1_idx = np.argmax(predictions)
                    top_1_score = predictions[top_1_idx] * 100
                    
                # Hiển thị kết quả
                st.success(f"🎉 Đây chắc chắn là: **{CLASS_NAMES[top_1_idx]}**")
                st.metric(label="Độ tự tin của AI", value=f"{top_1_score:.2f}%")
                
                if top_1_score >= 70.0:
                    st.balloons()
                
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
    2. Vẽ xong bấm **Phân tích**.
    3. Hệ thống sẽ tự động căn chỉnh tỷ lệ và làm phần việc còn lại.
    4. Bấm biểu tượng 🗑️ ở bảng vẽ để xóa và vẽ lại.
    """)
    st.divider()
    st.caption("🧠 Mô hình: CNN Deep Learning (Auto-Normalized)")
    st.caption("🔧 Tích hợp AI Auto-Crop bằng OpenCV")