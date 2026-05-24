import streamlit as st
import tensorflow as tf
import keras
import sys

# Cấu hình giao diện cơ bản để giữ app sống
st.set_page_config(page_title="KIỂM TRA PHIÊN BẢN", layout="centered")

st.title("📊 Kiểm Tra Hệ Sinh Thái Thư Viện")
st.write("Dưới đây là thông số môi trường chính xác đang chạy trên máy chủ Streamlit Cloud của m:")

st.divider()

# Hiển thị trực tiếp lên giao diện web bằng st.metric hoặc st.code
col1, col2 = st.columns(2)
with col1:
    st.metric(label="☘️ Phiên bản TensorFlow", value=tf.__version__)
with col2:
    st.metric(label="📦 Phiên bản Keras", value=keras.__version__)

st.divider()

# Hiện thêm chi tiết phiên bản Python để m nắm thông tin
st.info(f"🐍 **Phiên bản Python hiện tại của Server:** {sys.version}")

st.success("👉 Bây giờ m chỉ cần lấy đúng số phiên bản hiển thị ở trên màn hình này, quay về Colab cài đặt y hệt để xuất lại file model là xong!")