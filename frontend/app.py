import streamlit as st
import requests
import time
import base64
import matplotlib.pyplot as plt
import numpy as np

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Deepfake Detection (EfficientNet)", layout="centered")
st.title("Deepfake Detection (EfficientNet Only)")

# Sidebar: File upload and URL input
st.sidebar.header("Analyze Media")
file = st.sidebar.file_uploader("Upload image/video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
url = st.sidebar.text_input("Or enter media URL")

if st.sidebar.button("Analyze File") and file:
    with st.spinner("Analyzing file..."):
        files = {"file": (file.name, file, file.type)}
        resp = requests.post(f"{API_URL}/analyze/file", files=files)
        if resp.status_code == 200:
            result = resp.json()
            st.session_state['last_result'] = result
        else:
            st.error(f"Error: {resp.text}")

if st.sidebar.button("Analyze URL") and url:
    with st.spinner("Analyzing URL..."):
        resp = requests.post(f"{API_URL}/analyze/url", json={"url": url})
        if resp.status_code == 200:
            result = resp.json()
            st.session_state['last_result'] = result
        else:
            st.error(f"Error: {resp.text}")

# Main: Show last prediction result
if 'last_result' in st.session_state:
    result = st.session_state['last_result']
    st.subheader("Prediction Result")
    pred = result['prediction']
    st.write(f"**Is Fake:** {pred['is_fake']}")
    st.write(f"**Confidence:** {pred['confidence']:.2f}")
    st.write("**Model Breakdown:**")
    conf = pred['model_breakdown']['efficientnet']
    labels = ['Fake', 'Real']
    sizes = [conf, 1-conf]
    colors = ['#ff6666', '#66b3ff']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
    st.write("**Frames Analyzed:**", result['metrics']['frames_analyzed'])
    st.write("**Processing Time:**", result['metrics']['processing_time'])
    st.subheader("Input Preview")
    if result.get('thumbnail'):
        st.image(base64.b64decode(result['thumbnail']), use_container_width=True)
    st.subheader("Visualization")
    if result.get('visualization'):
        st.image(base64.b64decode(result['visualization']), use_container_width=True)

# Real-time metrics (fetch once per page load, with manual refresh)
st.subheader("Backend System Metrics")
if st.button("Refresh Metrics"):
    st.experimental_rerun()
try:
    resp = requests.get(f"{API_URL}/metrics")
    if resp.status_code == 200:
        metrics = resp.json()
        st.write(f"**Total Requests:** {metrics['total_requests']}")
        st.write(f"**Avg Processing Time:** {metrics['average_processing_time']:.2f}s")
        st.write(f"**Avg Frames Analyzed:** {metrics['average_frames_analyzed']:.2f}")
        st.write(f"**CPU Usage:** {metrics['system']['cpu_percent']}%")
        st.write(f"**Memory Usage:** {metrics['system']['memory_percent']}%")
    else:
        st.error(f"Error fetching metrics: {resp.text}")
except Exception as e:
    st.error(f"Exception: {e}") 