import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import cv2
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Cấu hình trang
st.set_page_config(page_title="AI NHẬN DIỆN HÌNH HỌC", page_icon="📐", layout="wide")

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

# Khai báo 16 nhãn - ĐÃ CẬP NHẬT THEO ĐÚNG THỨ TỰ THƯ MỤC ALPHABET TỪ FOLDER CỦA M
CLASS_NAMES = [
    "Hình bát giác (batgiac)", 
    "Hình bình hành (binhhanh)", 
    "Hình chữ nhật (chunhat)", 
    "Hình cửu giác (cuugiac)", 
    "Hình lục giác (lucgiac)", 
    "Hình ngũ giác (ngugiac)", 
    "Hình bán nguyệt (nuatron)", 
    "Hình oval (oval)",
    "Hình ngôi sao (sao)", 
    "Hình tam giác (tamgiac)", 
    "Hình thang (thang)", 
    "Hình thập giác (thapgiac)", 
    "Hình thất giác (thatgiac)", 
    "Hình thoi (thoi)", 
    "Hình tròn (tron)", 
    "Hình vuông (vuong)"
]

# Tải model (dùng cache để không bị load lại mỗi lần vẽ)
@st.cache_resource
def load_ai_model():
    # ĐÃ CẬP NHẬT TÊN FILE MODEL
    return load_model("mo_hinh_nhan_dang_hinh_hoc.keras") 

model = load_ai_model()

# Header
st.markdown("<div class='main-title'>📐 AI Nhận Diện 16 Loại Hình Học</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Vẽ một hình bất kỳ vào khung bên dưới, AI sẽ chẩn đoán real-time.</div>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<div class='custom-green-tag' style='margin-bottom: 15px;'>🖌️ Bảng Vẽ Tương Tác</div>", unsafe_allow_html=True)
    
    # Tạo bảng vẽ
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",  
        stroke_width=12,                      
        stroke_color="#FFFFFF",               
        background_color="#000000",           
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.markdown("<div class='custom-green-tag' style='margin-bottom: 15px;'>🧠 Kết Quả Từ AI</div>", unsafe_allow_html=True)
    
    # Xử lý ảnh realtime khi có nét vẽ
    if canvas_result.image_data is not None:
        img_array = canvas_result.image_data
        
        # Chuyển RGBA sang Grayscale
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
        
        # Chỉ dự đoán nếu canvas không trống
        if np.any(img_gray > 0):
            # 1. Resize về kích thước model yêu cầu (VD: 28x28)
            img_resized = cv2.resize(img_gray, (28, 28))
            
            # 2. Normalize về [0, 1]
            img_normalized = img_resized.astype('float32') / 255.0
            
            # 3. Reshape cho khớp model 
            # --- LƯU Ý DÒNG NÀY ---
            # NẾU M DÙNG MẠNG DENSE (Multi-Layer Perceptron):
            img_ready = img_normalized.reshape((1, 28 * 28))
            
            # NẾU M DÙNG MẠNG CNN (Conv2D): mở comment dòng dưới và đóng dòng trên lại
            # img_ready = img_normalized.reshape((1, 28, 28, 1))
            
            # 4. Dự đoán
            preds = model.predict(img_ready)[0]
            digit = np.argmax(preds)
            confidence = preds[digit] * 100
            
            st.metric(
                label="🎯 KẾT QUẢ PHÂN TÍCH", 
                value=f"{CLASS_NAMES[digit]}", 
                delta=f"Độ tin cậy: {confidence:.2f}%"
            )
            
            # Vẽ biểu đồ phân phối xác suất top 5
            df_preds = pd.DataFrame({
                'Loại hình': CLASS_NAMES,
                'Xác suất (%)': preds * 100
            }).sort_values(by='Xác suất (%)', ascending=True)
            
            top_5_df = df_preds.tail(5)
            
            fig = px.bar(
                top_5_df, x='Xác suất (%)', y='Loại hình',
                orientation='h', text_auto='.1f',
                color='Xác suất (%)', color_continuous_scale="Blues"
            )
            fig.update_layout(xaxis_title="", yaxis_title="", height=250, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Hãy vẽ gì đó vào bảng bên trái để AI phân tích nhé!")