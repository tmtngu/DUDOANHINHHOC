import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import cv2
import os
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Cấu hình trang ứng dụng
st.set_page_config(page_title="AI NHẬN DIỆN HÌNH HỌC", page_icon="📐", layout="wide")

# Áp dụng CSS tùy chỉnh giao diện
st.markdown("""
    <style>
    .main-title { font-size: 2.6rem; font-weight: 700; margin-bottom: 5px; }
    .sub-title { font-size: 1.1rem; margin-bottom: 25px; color: #666; }
    .custom-green-tag {
        display: inline-block; background-color: #22C55E !important; color: #FFFFFF !important;
        padding: 8px 24px; border-radius: 30px; font-weight: 600; font-size: 0.95rem;
    }
    </style>
""", unsafe_allow_html=True)

# Danh sách nhãn 16 loại hình học
CLASS_NAMES = [
    "Hình bát giác (batgiac)", "Hình bình hành (binhhanh)", "Hình chữ nhật (chunhat)",
    "Hình cửu giác (cuugiac)", "Hình lục giác (lucgiac)", "Hình ngũ giác (ngugiac)",
    "Hình bán nguyệt (nuatron)", "Hình oval (oval)", "Hình ngôi sao (sao)",
    "Hình tam giác (tamgiac)", "Hình thang (thang)", "Hình thập giác (thapgiac)",
    "Hình thất giác (thatgiac)", "Hình thoi (thoi)", "Hình tròn (tron)", "Hình vuông (vuong)"
]

# Hàm tải mô hình AI (Sử dụng bộ nhớ cache để không bị load lại mỗi lần vẽ)
@st.cache_resource
def load_my_model():
    model_path = "mo_hinh_nhan_dang_hinh_hoc.keras"
    if not os.path.exists(model_path):
        return None
    return load_model(model_path)

model = load_my_model()

# Tiêu đề ứng dụng
st.markdown("<div class='main-title'>📐 AI Nhận Diện 16 Loại Hình Học</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Vẽ một hình bất kỳ vào khung bên dưới, AI sẽ chẩn đoán real-time.</div>", unsafe_allow_html=True)

if model is None:
    st.error("❌ Chưa tìm thấy file 'mo_hinh_nhan_dang_hinh_hoc.keras' ở cùng thư mục với file app.py! Vui lòng kiểm tra lại.")
else:
    # Chia giao diện làm 2 cột
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("<div class='custom-green-tag' style='margin-bottom: 15px;'>🖌️ Bảng Vẽ Tương Tác</div>", unsafe_allow_html=True)
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)", stroke_width=12,
            stroke_color="#FFFFFF", background_color="#000000",
            height=300, width=300, drawing_mode="freedraw", key="canvas",
        )

    with col2:
        st.markdown("<div class='custom-green-tag' style='margin-bottom: 15px;'>🧠 Kết Quả Từ AI</div>", unsafe_allow_html=True)

        if canvas_result.image_data is not None:
            img_array = canvas_result.image_data
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)

            if np.any(img_gray > 0):
                # Xử lý ảnh về kích thước 64x64 phù hợp với model
                img_resized = cv2.resize(img_gray, (64, 64))
                img_normalized = img_resized.astype('float32') / 255.0

                # Reshape định dạng đầu vào CNN (batch_size, height, width, channels)
                img_ready = img_normalized.reshape((1, 64, 64, 1))

                # Dự đoán kết quả
                preds = model.predict(img_ready, verbose=0)[0]
                digit = np.argmax(preds)
                confidence = preds[digit] * 100

                # Hiển thị độ tin cậy kết quả cao nhất
                st.metric(label="🎯 KẾT QUẢ PHÂN TÍCH", value=f"{CLASS_NAMES[digit]}", delta=f"Độ tin cậy: {confidence:.2f}%")

                # Vẽ biểu đồ top 5 hình có xác suất cao nhất
                df_preds = pd.DataFrame({'Loại hình': CLASS_NAMES, 'Xác suất (%)': preds * 100}).sort_values(by='Xác suất (%)', ascending=True)
                fig = px.bar(df_preds.tail(5), x='Xác suất (%)', y='Loại hình', orientation='h', text_auto='.1f', color='Xác suất (%)', color_continuous_scale="Blues")
                fig.update_layout(xaxis_title="", yaxis_title="", height=250, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Hãy vẽ gì đó vào bảng bên trái để AI phân tích nhé!")