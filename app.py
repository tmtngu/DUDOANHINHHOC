import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN TRANG WEB
# ==========================================
st.set_page_config(page_title="AI Nhận Diện Hình Học Pro", page_icon="📐", layout="wide")

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
    .stRadio>div { gap: 15px; }
    </style>
""", unsafe_allow_html=True)

st.title("✨ HỆ THỐNG NHẬN DIỆN HÌNH HỌC AI (V3 - PRO) ✨")
st.markdown("<p style='text-align: center; font-size:18px;'>Tích hợp Bảng Vẽ Chuột & Bộ Tạo Hình Mẫu Bằng Thanh Trượt</p>", unsafe_allow_html=True)
st.divider()

# ==========================================
# 2. LOAD MODEL & DANH SÁCH NHÃN CHUẨN
# ==========================================
@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model("model_nhandienhinhhoc_v3.keras")

try:
    model = load_ai_model()
except Exception as e:
    st.error(f"⚠️ Lỗi không tìm thấy file mô hình: {e}\n\nÔng nhớ để file 'model_nhandienhinhhoc_v3.keras' chung thư mục với file app.py nhé!")
    st.stop()

# 16 Class của mô hình
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

# Map tên tiếng Việt sang từ khóa lập trình để vẽ hình mẫu chuẩn
SHAPE_GENERATOR_MAP = {
    "Hình ngôi sao": "sao",
    "Hình tam giác": "tamgiac",
    "Hình tròn": "tron",
    "Hình vuông": "vuong",
    "Hình chữ nhật": "chunhat",
    "Hình oval (Bầu dục)": "oval",
    "Hình thoi": "thoi",
    "Hình bình hành": "binhhanh",
    "Hình thang": "thang",
    "Hình ngũ giác (5 cạnh)": "ngugiac",
    "Hình lục giác (6 cạnh)": "lucgiac"
}

# ==========================================
# 3. HÀM TỰ ĐỘNG VẼ HÌNH HỌC CHUẨN THEO THANH TRƯỢT
# ==========================================
def draw_perfect_shape(shape_type, size, thickness):
    # Tạo ảnh nền trắng bóc kích thước 300x300
    img = np.full((300, 300), 255, dtype=np.uint8)
    center = (150, 150)
    
    if shape_type == "tron":
        cv2.circle(img, center, int(size), 0, thickness)
    elif shape_type == "vuong":
        s = int(size)
        cv2.rectangle(img, (150 - s, 150 - s), (150 + s, 150 + s), 0, thickness)
    elif shape_type == "chunhat":
        w, h = int(size * 1.3), int(size * 0.8)
        cv2.rectangle(img, (150 - w, 150 - h), (150 + w, 150 + h), 0, thickness)
    elif shape_type == "oval":
        w, h = int(size * 1.3), int(size * 0.8)
        cv2.ellipse(img, center, (w, h), 0, 0, 360, 0, thickness)
    elif shape_type == "tamgiac":
        r = int(size)
        pts = np.array([[150, 150 - r], [int(150 - r * 0.866), int(150 + r * 0.5)], [int(150 + r * 0.866), int(150 + r * 0.5)]], np.int32)
        cv2.polylines(img, [pts], True, 0, thickness)
    elif shape_type == "sao":
        r_out = int(size)
        r_in = int(size * 0.4)
        pts = []
        for i in range(10):
            r = r_out if i % 2 == 0 else r_in
            angle = i * np.pi / 5 - np.pi / 2
            pts.append([int(150 + r * np.cos(angle)), int(150 + r * np.sin(angle))])
        cv2.polylines(img, [np.array(pts, np.int32)], True, 0, thickness)
    elif shape_type == "thoi":
        w, h = int(size), int(size * 0.7)
        pts = np.array([[150, 150 - h], [150 + w, 150], [150, 150 + h], [150 - w, 150]], np.int32)
        cv2.polylines(img, [pts], True, 0, thickness)
    elif shape_type == "binhhanh":
        w, h, skew = int(size), int(size * 0.6), int(size * 0.3)
        pts = np.array([[150 - w + skew, 150 - h], [150 + w + skew, 150 - h], [150 + w - skew, 150 + h], [150 - w - skew, 150 + h]], np.int32)
        cv2.polylines(img, [pts], True, 0, thickness)
    elif shape_type == "thang":
        w_top, w_bot, h = int(size * 0.5), int(size * 1.0), int(size * 0.6)
        pts = np.array([[150 - w_top, 150 - h], [150 + w_top, 150 - h], [150 + w_bot, 150 + h], [150 - w_bot, 150 + h]], np.int32)
        cv2.polylines(img, [pts], True, 0, thickness)
    elif shape_type == "ngugiac":
        r = int(size)
        pts = [[int(150 + r * np.cos(i * 2 * np.pi / 5 - np.pi/2)), int(150 + r * np.sin(i * 2 * np.pi / 5 - np.pi/2))] for i in range(5)]
        cv2.polylines(img, [np.array(pts, np.int32)], True, 0, thickness)
    elif shape_type == "lucgiac":
        r = int(size)
        pts = [[int(150 + r * np.cos(i * np.pi / 3)), int(150 + r * np.sin(i * np.pi / 3))] for i in range(6)]
        cv2.polylines(img, [np.array(pts, np.int32)], True, 0, thickness)
        
    return img

# ==========================================
# 4. THIẾT KẾ GIAO DIỆN CHÍNH
# ==========================================
st.subheader("🛠️ Bước 1: Chọn phương thức tạo hình")
app_mode = st.radio(
    "Lựa chọn cách thức:",
    ("📐 Kéo thanh trượt tạo hình mẫu có sẵn (Khuyên dùng - Chuẩn 100%)", "🖌️ Tự vẽ bằng chuột / Kéo thả hình tự do trên bảng vẽ"),
    horizontal=True,
    label_visibility="collapsed"
)
st.divider()

col1, col2 = st.columns([4, 6], gap="large")

# Khởi tạo biến chứa ảnh xám cuối cùng trước khi đưa vào AI
final_gray_input = None

with col1:
    if "📐 Kéo" in app_mode:
        st.subheader("📐 Bảng điều khiển thanh trượt")
        
        selected_shape_vn = st.selectbox("1. Chọn hình muốn test:", list(SHAPE_GENERATOR_MAP.keys()))
        selected_size = st.slider("2. Kéo điều chỉnh Kích thước (Size):", min_value=30, max_value=120, value=80, step=5)
        selected_thickness = st.slider("3. Kéo điều chỉnh Độ dày nét vẽ (Thickness):", min_value=2, max_value=8, value=3, step=1)
        
        # Gọi hàm vẽ tự động hình mẫu chuẩn
        internal_key = SHAPE_GENERATOR_MAP[selected_shape_vn]
        generated_img = draw_perfect_shape(internal_key, selected_size, selected_thickness)
        
        st.write("**🖼️ Xem trước hình mẫu chuẩn:**")
        st.image(generated_img, width=300, caption="Hình sinh ra tự động từ ma trận máy tính")
        final_gray_input = generated_img
        
    else:
        st.subheader("🖌️ Bảng vẽ tương tác")
        canvas_tool = st.radio(
            "Công cụ vẽ:", 
            ("🖌️ Vẽ tự do", "⭕ Kéo thả hình Tròn", "🟦 Kéo thả hình Chữ nhật", "📏 Đoạn thẳng"),
            horizontal=True
        )
        
        canvas_mode_map = {
            "🖌️ Vẽ tự do": "freedraw",
            "⭕ Kéo thả hình Tròn": "circle",
            "🟦 Kéo thả hình Chữ nhật": "rect",
            "📏 Đoạn thẳng": "line"
        }
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)", 
            stroke_width=3, 
            stroke_color="#000000", 
            background_color="#FFFFFF", 
            height=300,
            width=300,
            drawing_mode=canvas_mode_map[canvas_tool],
            key="canvas_pro",
        )

with col2:
    st.subheader("🤖 Kết quả phân tích từ AI")
    st.caption("Bấm nút phân tích để kích hoạt não bộ AI.")
    
    if st.button("🚀 KÍCH HOẠT AI PHÂN TÍCH HÌNH HỌC", use_container_width=True):
        
        # Lấy dữ liệu ảnh dựa theo chế độ đang chọn
        if "📐 Kéo" in app_mode:
            # Chế độ thanh trượt đã có ảnh sẵn
            gray_image = final_gray_input
        else:
            # Chế độ vẽ tay lấy từ canvas kết quả
            if canvas_result.image_data is not None:
                img_rgba = canvas_result.image_data
                if np.all(img_rgba[:, :, 3] == 0):
                    gray_image = None
                else:
                    img_pil = Image.fromarray(img_rgba.astype('uint8'), 'RGBA')
                    white_bg = Image.new("RGB", img_pil.size, (255, 255, 255))
                    white_bg.paste(img_pil, mask=img_pil.split()[3])
                    gray_image = np.array(white_bg.convert("L"))
            else:
                gray_image = None
                
        # --- TIẾN HÀNH XỬ LÝ ẢNH CHUẨN HÓA CHO AI ---
        if gray_image is None:
            st.warning("⚠️ Không tìm thấy dữ liệu nét vẽ! Ông nhớ vẽ gì đó hoặc chọn hình mẫu trước khi bấm phân tích nhé!")
        else:
            with st.spinner("🧠 AI đang phân tích cấu trúc hình học..."):
                # 1. AUTO-CROP CẮT BỎ RÌA TRẮNG
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
                    
                    # 2. SQUARE PADDING (TẠO KHUNG VUÔNG CHỐNG MÉO HÌNH)
                    max_side = max(w, h)
                    square_img = np.full((max_side, max_side), 255, dtype=np.uint8)
                    offset_x = (max_side - w) // 2
                    offset_y = (max_side - h) // 2
                    square_img[offset_y:offset_y+h, offset_x:offset_x+w] = cropped_image
                    
                    # 3. ÉP TRẮNG ĐEN TUYỆT ĐỐI TRƯỚC KHI RESIZE ĐỂ GIỮ NÉT
                    _, binarized_thick = cv2.threshold(square_img, 200, 255, cv2.THRESH_BINARY)
                    
                    # 4. THU NHỎ XUỐNG KÍCH THƯỚC MÔ HÌNH HỌC (64x64)
                    final_image = cv2.resize(binarized_thick, (64, 64), interpolation=cv2.INTER_AREA)
                else:
                    final_image = cv2.resize(gray_image, (64, 64), interpolation=cv2.INTER_AREA)
                
                # 5. ĐẨY VÀO MÔ HÌNH DỰ ĐOÁN (KHÔNG CHIA 255)
                input_tensor = np.expand_dims(final_image, axis=[0, -1])
                predictions = model.predict(input_tensor)[0]
                
                top_1_idx = np.argmax(predictions)
                top_1_score = predictions[top_1_idx] * 100
                
            # --- IN KẾT QUẢ RA MÀN HÌNH WEB ---
            st.success(f"🎉 Kết quả dự đoán: **{CLASS_NAMES[top_1_idx]}**")
            st.metric(label="Độ chính xác / tự tin của AI", value=f"{top_1_score:.2f}%")
            
            # Hiển thị camera ẩn sát sườn
            st.image(final_image, caption="🔍 Ảnh thu nhỏ thực tế (64x64) nạp vào não AI", width=120)
            
            if top_1_score >= 80.0:
                st.balloons()
            
            st.markdown("#### 📊 Top 3 khả năng cao nhất:")
            top_3_indices = np.argsort(predictions)[-3:][::-1]
            for idx in top_3_indices:
                score = predictions[idx] * 100
                st.write(f"**{CLASS_NAMES[idx]}**: {score:.1f}%")
                st.progress(int(score))