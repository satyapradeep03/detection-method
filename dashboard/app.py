import streamlit as st
import requests
import time
import base64
import json

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Deepfake Detection Dashboard", layout="wide")
st.title("Deepfake Detection Analytics Dashboard")

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
    st.bar_chart(pred['model_breakdown'])
    st.write("**Frames Analyzed:**", result['metrics']['frames_analyzed'])
    st.write("**Processing Time:**", result['metrics']['processing_time'])
    st.subheader("Visualization")
    if result['visualization']:
        st.image(base64.b64decode(result['visualization']), use_column_width=True)

# Real-time metrics
st.subheader("System & Model Metrics (updates every 2s)")
metrics_placeholder = st.empty()

while True:
    try:
        resp = requests.get(f"{API_URL}/metrics")
        if resp.status_code == 200:
            metrics = resp.json()
            with metrics_placeholder.container():
                st.write(f"**Total Requests:** {metrics['total_requests']}")
                st.write(f"**Avg Processing Time:** {metrics['average_processing_time']:.2f}s")
                st.write(f"**Avg Frames Analyzed:** {metrics['average_frames_analyzed']:.2f}")
                st.write(f"**CPU Usage:** {metrics['system']['cpu_percent']}%")
                st.write(f"**Memory Usage:** {metrics['system']['memory_percent']}%")
                if 'gpu_memory_allocated_MB' in metrics['system']:
                    st.write(f"**GPU Memory Allocated:** {metrics['system']['gpu_memory_allocated_MB']:.1f} MB")
                    st.write(f"**GPU Memory Reserved:** {metrics['system']['gpu_memory_reserved_MB']:.1f} MB")
        else:
            metrics_placeholder.error(f"Error fetching metrics: {resp.text}")
    except Exception as e:
        metrics_placeholder.error(f"Exception: {e}")
    time.sleep(2) 