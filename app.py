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

st.title("✨ HỆ THỐNG NHẬN DIỆN HÌNH HỌC AI (V3 - DEBUG PRO) ✨")
st.divider()

# ==========================================
# 2. THANH CẤU HÌNH SIDEBAR ĐỂ DÒ TẦN SỐ AI
# ==========================================
st.sidebar.header("⚙️ BỘ CẤU HÌNH CHUẨN HÓA AI")
st.sidebar.markdown("Ông hãy điều chỉnh các nút này để khớp với lúc ông train model trên Colab nhé!")

# Cấu hình 1: Kích thước ảnh đầu vào
size_option = st.sidebar.selectbox(
    "1. Kích thước ảnh nạp vào AI:",
    ("224x224 (Khuyên dùng theo Dataset)", "64x64"),
    index=0
)
TARGET_SIZE = 224 if "224x224" in size_option else 64

# Cấu hình 2: Bật tắt Auto-Crop
USE_AUTOCROP = st.sidebar.checkbox(
    "2. Cắt ảnh sát viền (Auto-Crop)", 
    value=False
)
st.sidebar.caption("💡 *Lời khuyên:* Nên TẮT vì tập dữ liệu mẫu của ông luôn giữ nguyên khoảng trắng bao quanh hình vẽ.")

# Cấu hình 3: Chuẩn hóa chia 255
USE_RESCOLED = st.sidebar.checkbox(
    "3. Chuẩn hóa dữ liệu (Chia ảnh cho 255.0)", 
    value=True
)
st.sidebar.caption("💡 *Lời khuyên:* Nên THỬ CẢ HAI (Bật hoặc Tắt) để xem lúc train trên Colab ông có xử lý bước này không.")


# ==========================================
# 3. LOAD MODEL VÀ DANH SÁCH CLASS
# ==========================================
@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model("model_nhandienhinhhoc_v3.keras")

try:
    model = load_ai_model()
except Exception as e:
    st.error(f"⚠️ Lỗi không tìm thấy file mô hình: {e}")
    st.stop()

CLASS_NAMES = [
    "Hình bát giác (8 cạnh)", "Hình bình hành", "Hình chữ nhật", "Hình cửu giác (9 cạnh)",
    "Hình lục giác (6 cạnh)", "Hình ngũ giác (5 cạnh)", "Hình bán nguyệt", "Hình oval (Bầu dục)",
    "Hình ngôi sao", "Hình tam giác", "Hình thang", "Hình thập giác (10 cạnh)",
    "Hình thất giác (7 cạnh)", "Hình thoi", "Hình tròn", "Hình vuông"
]

SHAPE_GENERATOR_MAP = {
    "Hình tròn": "tron", "Hình vuông": "vuong", "Hình chữ nhật": "chunhat",
    "Hình oval (Bầu dục)": "oval", "Hình ngôi sao": "sao", "Hình tam giác": "tamgiac",
    "Hình thoi": "thoi", "Hình bình hành": "binhhanh", "Hình thang": "thang",
    "Hình ngũ giác (5 cạnh)": "ngugiac", "Hình lục giác (6 cạnh)": "lucgiac"
}

# ==========================================
# 4. HÀM TẠO HÌNH MẪU CHUẨN (Cân đối như tập Dataset)
# ==========================================
def draw_perfect_shape(shape_type, size, thickness):
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
# 5. THIẾT KẾ GIAO DIỆN CHÍNH
# ==========================================
st.subheader("🛠️ Bước 1: Chọn phương thức tạo hình")
app_mode = st.radio(
    "Lựa chọn cách thức:",
    ("📐 Kéo thanh trượt tạo hình mẫu có sẵn (Chuẩn như Dataset)", "🖌️ Tự vẽ bằng chuột / Kéo thả hình tự do"),
    horizontal=True,
    label_visibility="collapsed"
)
st.divider()

col1, col2 = st.columns([4, 6], gap="large")
final_gray_input = None

with col1:
    if "📐 Kéo" in app_mode:
        st.subheader("📐 Bộ tạo hình mẫu")
        selected_shape_vn = st.selectbox("1. Chọn hình muốn kiểm tra:", list(SHAPE_GENERATOR_MAP.keys()))
        selected_size = st.slider("2. Kích thước hình (Size):", min_value=30, max_value=120, value=70, step=5)
        selected_thickness = st.slider("3. Độ dày nét vẽ (Thickness):", min_value=1, max_value=5, value=2, step=1)
        
        internal_key = SHAPE_GENERATOR_MAP[selected_shape_vn]
        generated_img = draw_perfect_shape(internal_key, selected_size, selected_thickness)
        
        st.write("**🖼️ Hình ảnh thực tế trên Canvas:**")
        st.image(generated_img, width=300)
        final_gray_input = generated_img
    else:
        st.subheader("🖌️ Khu vực vẽ tay")
        canvas_tool = st.radio("Công cụ:", ("🖌️ Vẽ tự do", "⭕ Hình Tròn", "🟦 Hình Chữ nhật"), horizontal=True)
        canvas_mode_map = {"🖌️ Vẽ tự do": "freedraw", "⭕ Hình Tròn": "circle", "🟦 Hình Chữ nhật": "rect"}
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)", stroke_width=3, stroke_color="#000000",
            background_color="#FFFFFF", height=300, width=300,
            drawing_mode=canvas_mode_map[canvas_tool], key="canvas_pro_v3",
        )

with col2:
    st.subheader("🤖 Kết quả phân tích từ AI")
    
    if st.button("🚀 KÍCH HOẠT AI PHÂN TÍCH HÌNH HỌC", use_container_width=True):
        if "📐 Kéo" in app_mode:
            gray_image = final_gray_input
        else:
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
                
        if gray_image is None:
            st.warning("⚠️ Không tìm thấy nét vẽ! Vui lòng thao tác trước khi phân tích.")
        else:
            with st.spinner("🧠 AI đang xử lý cấu trúc dữ liệu..."):
                
                # CHẾ ĐỘ 1: NẾU BẬT AUTO-CROP SÁT VIỀN
                if USE_AUTOCROP:
                    _, thresh = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY_INV)
                    coords = cv2.findNonZero(thresh)
                    if coords is not None:
                        x, y, w, h = cv2.boundingRect(coords)
                        pad = 15
                        x = max(0, x - pad); y = max(0, y - pad)
                        w = min(gray_image.shape[1] - x, w + 2*pad)
                        h = min(gray_image.shape[0] - y, h + 2*pad)
                        cropped_image = gray_image[y:y+h, x:x+w]
                        
                        max_side = max(w, h)
                        square_img = np.full((max_side, max_side), 255, dtype=np.uint8)
                        offset_x = (max_side - w) // 2; offset_y = (max_side - h) // 2
                        square_img[offset_y:offset_y+h, offset_x:offset_x+w] = cropped_image
                        process_target = square_img
                    else:
                        process_target = gray_image
                else:
                    # CHẾ ĐỘ 2: GIỮ NGUYÊN KHÔNG GIAN ĐỆM XUNG QUANH HÌNH (GIỐNG DATASET)
                    process_target = gray_image

                # Ép binarize giữ độ sắc nét và resize đúng kích thước mục tiêu
                _, binarized = cv2.threshold(process_target, 220, 255, cv2.THRESH_BINARY)
                final_image = cv2.resize(binarized, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
                
                # Chuyển đổi kiểu dữ liệu và thực hiện chuẩn hóa pixel nếu bật tính năng
                input_array = final_image.astype(np.float32)
                if USE_RESCOLED:
                    input_array = input_array / 255.0
                
                # Tạo tensor đầu vào (Batch_size, Height, Width, Channels)
                input_tensor = np.expand_dims(input_array, axis=[0, -1])
                predictions = model.predict(input_tensor)[0]
                
                top_1_idx = np.argmax(predictions)
                top_1_score = predictions[top_1_idx] * 100
                
            # --- HIỂN THỊ KẾT QUẢ ---
            st.success(f"🎉 Kết quả dự đoán: **{CLASS_NAMES[top_1_idx]}**")
            st.metric(label="Độ tự tin của AI", value=f"{top_1_score:.2f}%")
            
            st.image(final_image, caption=f"🔍 Ảnh thực tế ({TARGET_SIZE}x{TARGET_SIZE}) nạp vào não AI", width=140)
            
            if top_1_score >= 80.0:
                st.balloons()
            
            st.markdown("#### 📊 Top 3 khả năng cao nhất:")
            top_3_indices = np.argsort(predictions)[-3:][::-1]
            for idx in top_3_indices:
                score = predictions[idx] * 100
                st.write(f"**{CLASS_NAMES[idx]}**: {score:.1f}%")
                st.progress(int(score))