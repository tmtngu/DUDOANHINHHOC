import streamlit as st
import tensorflow as tf
import keras

# 🚨 ÉP APP IN PHIÊN BẢN RA LOG ĐỂ MÌNH XEM
print("=== PHIÊN BẢN TRÊN STREAMLIT CLOUD ===")
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("=======================================")

st.write("Đang kiểm tra phiên bản...")