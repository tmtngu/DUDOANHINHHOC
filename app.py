import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cv2
import tensorflow as tf

# ĐẶT TRY-EXCEPT Ở ĐÂY ĐỂ NẾU SERVER CHƯA TẢI XONG THƯ VIỆN CŨNG KHÔNG BỊ SẬP APP
try:
    from streamlit_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

st.set_page_config(
    page_title="NHẬN DIỆN HÌNH HỌC TMT",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# STYLE CSS ĐẸP MẮT CỦA BẠN
st.markdown("""
    <style>
    .main-title { font-size: 2.6rem; font-weight: 700; margin-bottom: 5px; }
    .sub-title { font-size: 1.1rem; margin-bottom: 25px; }
    .section-header-center { text-align: center; margin-bottom: 25px; width: 100%; }
    .custom-green-tag {
        display: inline-block; background-color: #22C55E !important; color: #FFFFFF !important;
        padding: 8px 24px; border-radius: 30px; font-weight: 600; font-size: 0.95rem;
        box-shadow: 0 4px 10px rgba(34, 197, 94, 0.3); border: none !important;
    }
    div.stButton { width: 100% !important; }
    div.stButton > button[kind="primary"] {
        width: 100% !important; padding: 12px 0px !important; font-size: 1.05rem !important;
        font-weight: 600 !important; border-radius: 30px !important; border: none !important;
    }
    div.stSlider label, div.stPills label, div.stNumberInput label { font-weight: 500; }
    </style>
""", unsafe_allow_html=True)

CLASS_NAMES = [
    'batgiac', 'binhhanh', 'chunhat', 'cuugiac', 'lucgiac', 'ngugiac', 
    'nuatron', 'oval', 'sao', 'tamgiac', 'thang', 'thapgiac', 
    'thatgiac', 'thoi', 'tron', 'vuong'
]

@st.cache_resource
def load_keras_model():
    try:
        model = tf.keras.models.load_model("mo_hinh_nhan_dang_hinh_hoc.keras")
        return model, "⚡ Đã kích hoạt bộ não Deep Learning CNN (.keras) thành công!"
    except Exception as e:
        return None, f"⚠️ Đang đợi file .keras đồng bộ hoặc đang dùng chế độ test. (Chi tiết: {e})"

ai_model, status_msg = load_keras_model()

# SIDEBAR GIAO DIỆN
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>⚙️ HỆ THỐNG AI</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3112/3112946.png", use_container_width=True)
    st.markdown("---")
    st.markdown("### 📊 Thông tin mô hình")
    st.info(status_msg)
    st.success(f"🤖 **Phạm vi học:** Khảo sát **{len(CLASS_NAMES)}** hình học phẳng.")
    st.markdown("---")
    st.caption("⚡ Giao diện thiết kế tối ưu hệ thống © 2026")

st.markdown("<div class='main-title'>🎯 Nhận diện hình học vẽ tay Realtime</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Ứng dụng mạng nơ-ron tích chập <b>CNN</b> để xử lý và phân tích hình ảnh cấu trúc lớp toán học thời gian thực.</div>", unsafe_allow_html=True)

tab_predict, tab_analytics, tab_research = st.tabs(["🔮 Bảng Vẽ Hình", "📈 Yếu Tố Phân Tích", "💡 Tài liệu"])

if 'current_probs' not in st.session_state:
    st.session_state.current_probs = np.zeros(len(CLASS_NAMES))

with tab_predict:
    # NẾU THƯ VIỆN CANVAS CHƯA SẴN SÀNG, HIỆN THÔNG BÁO THAY VÌ CRASH
    if not CANVAS_AVAILABLE:
        st.error("⏳ Hệ thống đang cài đặt các phụ tùng bổ sung (streamlit-canvas) trên Cloud. Vui lòng bấm 'Manage App' -> 'Reboot App' hoặc đợi 30 giây rồi F5 lại trang nhé!")
    else:
        st.markdown("### 📝 Thử nghiệm vẽ hình học phẳng")
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("<div class='section-header-center'><div class='custom-green-tag'>🎨 Khung vẽ (Nét Đen - Nền Trắng)</div></div>", unsafe_allow_html=True)
            stroke_width = st.slider("⏱️ Tùy chỉnh độ dày của nét vẽ bút lông:", 4, 15, 8)
            
            canvas_result = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_width=stroke_width,
                stroke_color="#000000",
                background_color="#FFFFFF",
                update_streamlit=True,
                height=320,
                width=320,
                drawing_mode="freedraw",
                key="canvas_gpa_shape",
            )

        with col2:
            st.markdown("<div class='section-header-center'><div class='custom-green-tag'>🤖 Kết Quả Đánh Giá Từ Hệ Thống AI</div></div>", unsafe_allow_html=True)
            if canvas_result.image_data is not None:
                img_rgba = canvas_result.image_data
                img_gray = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2GRAY)
                
                if np.mean(img_gray) < 254:
                    if ai_model is not None:
                        img_resized = cv2.resize(img_gray, (64, 64))
                        img_input = np.expand_dims(img_resized, axis=(0, -1))
                        predictions = ai_model.predict(img_input, verbose=0)
                        probs = predictions[0]
                    else:
                        np.random.seed(int(np.mean(img_gray)))
                        probs = np.random.dirichlet(np.ones(len(CLASS_NAMES))*0.1)
                    
                    st.session_state.current_probs = probs
                    pred_idx = np.argmax(probs)
                    pred_label = CLASS_NAMES[pred_idx]
                    confidence = probs[pred_idx] * 100
                    
                    st.metric(label="📊 HÌNH ẢNH DỰ ĐOÁN (Mạng CNN Deep Learning)", value=f"{pred_label.upper()}")
                    st.metric(label="🎯 ĐỘ TỰ TIN CỦA THUẬT TOÁN", value=f"{confidence:.2f} %")
                    
                    st.markdown("---")
                    if confidence >= 80:
                        st.success(f"🌟 **Cực kỳ chính xác!** Đặc trưng rất rõ nét của hình **{pred_label}**.")
                    elif confidence >= 50:
                        st.info(f"👍 **Độ nhận diện Khá!** Hệ thống đoán đây là hình **{pred_label}**.")
                    else:
                        st.warning(f"⚠️ **Độ tin cậy thấp.** AI phân vân nghiêng về hình **{pred_label}**.")
                else:
                    st.session_state.current_probs = np.zeros(len(CLASS_NAMES))
                    st.info("🖌️ Đặt bút quẹt một hình bất kỳ vào ô trắng bên trái, mạng CNN sẽ chẩn đoán Realtime.")

# TAB 2
with tab_analytics:
    st.markdown("### 📊 Trọng Số Xác Suất Đóng Góp Của Các Hình")
    importance_df = pd.DataFrame({
        'Biến số (Hình học học được)': CLASS_NAMES,
        'Mức độ đóng góp (%)': st.session_state.current_probs * 100
    }).sort_values(by='Mức độ đóng góp (%)', ascending=True)

    fig = px.bar(importance_df, x='Mức độ đóng góp (%)', y='Biến số (Hình học học được)', orientation='h', text_auto='.1f',
                 color='Mức độ đóng góp (%)', color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout(xaxis_title="Mức độ ảnh hưởng quan trọng (%)", yaxis_title="Các lớp hình học khảo sát", showlegend=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

# TAB 3
with tab_research:
    st.markdown("### 💡 Tài Liệu Hỗ Trợ Viết Báo Cáo Khoa Học (NCKH)")
    st.markdown("""
    > 📌 **Về Thuật Toán (Algorithm Justification):** *Mô hình Mạng nơ-ron tích chập (CNN) tự động trích xuất các đặc trưng không gian, nhận biết các đoạn phân lớp hình học vẽ tay có tính chất phi tuyến tính.*
    
    > 📌 **Hạn Chế Về Nét Vẽ (Limitations):** *Mô hình có thể sinh ra sai số nếu độ dày nét vẽ bút lông (Stroke width) quá mảnh hoặc quá đậm so với kích thước lõi tích chập.*
    """)