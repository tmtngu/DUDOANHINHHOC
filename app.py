import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import plotly.express as px

# 1. CẤU HÌNH TRANG VÀ CSS (Chuẩn Form ông đưa)
st.set_page_config(
    page_title="DỰ ĐOÁN HÌNH HỌC AI",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 2.6rem; font-weight: 700; margin-bottom: 5px;
    }
    .sub-title {
        font-size: 1.1rem; margin-bottom: 25px; color: #555;
    }
    .section-header-center {
        text-align: center; margin-bottom: 20px; width: 100%;
    }
    .custom-green-tag {
        display: inline-block; background-color: #22C55E !important; color: #FFFFFF !important;
        padding: 8px 24px; border-radius: 30px; font-weight: 600; font-size: 0.95rem;
        box-shadow: 0 4px 10px rgba(34, 197, 94, 0.3); border: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. LOAD MODEL & DATA
CLASSES = ['Bình Hành', 'Chữ Nhật', 'Ngôi Sao', 'Tam Giác', 'Đường Thẳng', 'Hình Tròn', 'Hình Vuông']

@st.cache_resource
def load_ai_model():
    try:
        model = tf.keras.models.load_model('Model_7Hinh_VuaVan.keras')
        status = "Trực tuyến (Sẵn sàng)"
        return model, status
    except:
        return None, "Lỗi: Không tìm thấy file model!"

model, model_status = load_ai_model()

# 3. SIDEBAR
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>⚙️ HỆ THỐNG AI</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103130.png", use_container_width=True) # Đổi icon AI hình học
    st.markdown("---")
    st.markdown("### 📊 Thông tin Mô hình")
    st.info("📁 **Dữ liệu huấn luyện:**\n10,500 mẫu (500x3 mỗi class)")
    if model:
        st.success(f"🤖 **Trạng thái:**\n{model_status}")
    else:
        st.error(f"🚨 **Trạng thái:**\n{model_status}")
    
    st.markdown("---")
    st.caption("⚡ Thiết kế nhằm mục đích áp dụng kiến thức AI © 2026")

# 4. HEADER
st.markdown("<div class='main-title'>🎯 Hệ Thống AI Nhận Diện Hình Học </div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Ứng dụng mạng nơ-ron tích chập (CNN) để phân loại 7 hình học cơ bản theo thời gian thực.</div>", unsafe_allow_html=True)

# 5. TABS
tab_predict, tab_analytics, tab_research = st.tabs([
    "🎨 Bảng Vẽ Nhận Diện",
    "📈 Biểu Đồ Xác Suất",
    "💡 Tài Liệu Kỹ Thuật"
])

# BIẾN TOÀN CỤC ĐỂ LƯU KẾT QUẢ CHO TAB 2
prediction_data = None

# TAB 1: BẢNG VẼ
with tab_predict:
    st.write("Vui lòng vẽ một hình khối bất kỳ vào bảng bên dưới, AI sẽ tự động phân tích và đưa ra kết quả:")
    
    col1, col2 = st.columns([1.5, 1], gap="large")

    with col1:
        st.markdown("""
            <div class='section-header-center'>
                <div class='custom-green-tag'>✏️ Không gian vẽ tay</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Tạo canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)", stroke_width=8,
            stroke_color="#000000", background_color="#FFFFFF",
            height=400, width=100, # width 100% của cột
            drawing_mode="freedraw", key="canvas_shape"
        )
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            st.caption("💡 Mẹo: Vẽ nét liền và khép kín để AI nhận diện tốt nhất.")

    with col2:
        st.markdown("""
            <div class='section-header-center'>
                <div class='custom-green-tag'>🧠 Kết quả từ AI</div>
            </div>
        """, unsafe_allow_html=True)
        
        if canvas_result.image_data is not None and model is not None:
            img = cv2.cvtColor(canvas_result.image_data, cv2.COLOR_RGBA2GRAY)
            
            if np.any(img < 255): # Có nét vẽ
                # Preprocessing ảnh (Cắt và đưa về 224x224)
                pts = np.argwhere(img < 255)
                y_min, x_min = pts[:,0].min(), pts[:,1].min()
                y_max, x_max = pts[:,0].max(), pts[:,1].max()
                cropped = img[y_min:y_max+1, x_min:x_max+1]
                
                h, w = cropped.shape
                side = max(h, w) + 40
                pad = np.ones((side, side), dtype=np.uint8) * 255
                pad[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w] = cropped
                
                input_img = cv2.resize(pad, (224, 224))
                input_img_rgb = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)
                
                # Predict
                pred = model.predict(np.expand_dims(input_img_rgb, axis=0), verbose=0)[0]
                idx = np.argmax(pred)
                max_prob = pred[idx] * 100
                
                # Lưu data cho Tab 2
                prediction_data = pd.DataFrame({
                    'Hình khối': CLASSES,
                    'Xác suất (%)': pred * 100
                }).sort_values(by='Xác suất (%)', ascending=True)

                # Hiển thị Metric
                st.metric(label="DỰ ĐOÁN HÌNH DÁNG", value=CLASSES[idx], delta=f"Độ tin cậy: {max_prob:.2f}%")
                
                # Đánh giá độ tin cậy
                if max_prob >= 90:
                    st.success("🌟 **Cực kỳ rõ ràng!** Nét vẽ rất chuẩn xác, AI không hề do dự khi nhận diện hình này.")
                elif max_prob >= 60:
                    st.warning("⚠️ **Có chút phân vân.** Hình vẽ hơi méo hoặc giống với một hình khác. Hãy kiểm tra biểu đồ xác suất ở Tab 2 nhé.")
                else:
                    st.error("🚨 **Khó nhận diện!** Nét vẽ quá rối hoặc không nằm trong tập dữ liệu 7 hình cơ bản. Thử vẽ lại rõ nét hơn xem sao.")
            else:
                st.info("👈 Bảng đang trống. Hãy múa chuột vẽ gì đó đi!")

# TAB 2: BIỂU ĐỒ XÁC SUẤT
with tab_analytics:
    st.markdown("### 📊 Mức Độ Tự Tin Của Thuật Toán (Confidence Logic)")
    st.write("Biểu đồ này bóc tách thuật toán bên trong, cho thấy AI đang đánh giá xác suất hình vẽ của bạn thuộc về lớp (class) nào cao nhất.")

    if prediction_data is not None:
        fig = px.bar(
            prediction_data,
            x='Xác suất (%)',
            y='Hình khối',
            orientation='h',
            text_auto='.2f',
            color='Xác suất (%)',
            color_continuous_scale=px.colors.sequential.Greens # Tone xanh lá cho hợp UI
        )
        fig.update_layout(
            xaxis_title="Xác suất dự đoán (%)",
            yaxis_title="Các nhãn hình học",
            showlegend=False,
            height=450,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("🔍 Xem mảng xác suất thô (Raw Array)"):
            st.dataframe(
                prediction_data.sort_values(by='Xác suất (%)', ascending=False)
                .style.background_gradient(cmap='Greens'), 
                use_container_width=True
            )
    else:
        st.info("Vui lòng vẽ hình ở Tab 'Bảng Vẽ Nhận Diện' để xem phân tích biểu đồ.")

# TAB 3: TÀI LIỆU
with tab_research:
    st.markdown("### 💡 Tài Liệu Hỗ Trợ Viết Báo Cáo Khoa Học (NCKH)")
    st.write("Dùng các đoạn văn mẫu học thuật này để hỗ trợ viết phần Thảo luận (Discussion) & Phương pháp (Methodology):")

    st.markdown("""
    > 📌 **Về Thuật Toán (Algorithm Justification):** *Trong nghiên cứu Thị giác Máy tính (Computer Vision), mô hình Convolutional Neural Network (CNN) được lựa chọn nhờ khả năng tự động trích xuất các đặc trưng không gian (spatial features) từ hình ảnh. Mô hình xử lý tốt sự sai lệch do nét vẽ tay người dùng nhờ các lớp MaxPooling, giúp giảm thiểu hiện tượng quá khớp (overfitting).*

    > 📌 **Kỹ Thuật Xử Lý Tiền Dữ Liệu (Preprocessing Strategy):** *Thay vì đưa toàn bộ khung tranh trống vào mô hình gây nhiễu, hệ thống áp dụng thuật toán Bounding Box bằng OpenCV để cắt sát (crop) nét vẽ của người dùng. Hình ảnh sau đó được đệm (padding) thành hình vuông và thay đổi kích thước về ma trận 224x224 pixels, tối ưu hóa quá trình tính toán của mạng CNN.*

    > 📌 **Hạn Chế Về Sai Số Nhận Diện (Limitations & Edge Cases):** *Do dữ liệu huấn luyện chủ yếu tập trung vào 7 hình học tĩnh, hệ thống có thể gặp giới hạn (False Positives) khi người dùng cố tình vẽ các hình thù phức tạp (như động vật, chữ viết) nhưng mô hình vẫn ép xác suất vào 1 trong 7 lớp đã học (Softmax bottleneck).*
    """)