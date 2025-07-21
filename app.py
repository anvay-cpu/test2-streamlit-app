import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# DepthAnything imports (ensure proper Python path/setup)
from depth_anything_v2.dpt import DepthAnythingV2

# Device selection
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models only once (cached)
@st.cache_resource
def load_depth_model():
    model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    state_dict = torch.load('checkpoints/depth_anything_v2_vits.pth', map_location=DEVICE)
    model.load_state_dict(state_dict)
    return model.to(DEVICE).eval()

@st.cache_resource
def load_yolo_model():
    model = YOLO('yolov8n.pt')
    return model

def get_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def run_depth(frame, depth_model):
    with torch.no_grad():
        # Depth inference (expects BGR frame)
        depth = depth_model.infer_image(frame)
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype('uint8')
        depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
        # Resize colormap if needed
        if depth_colormap.shape[:2] != frame.shape[:2]:
            depth_colormap = cv2.resize(depth_colormap, (frame.shape[1], frame.shape[0]))
    combined = cv2.hconcat([frame, depth_colormap])
    return combined

def run_yolo(frame, yolo_model):
    # YOLO inference
    results = yolo_model(frame)
    # Visualized frame
    annotated_frame = results[0].plot()
    return annotated_frame

# Streamlit UI
st.title("Real-Time Depth & Object Detection")
mode = st.sidebar.selectbox("Select mode", ["Depth Estimation", "Object Detection (YOLOv8)"])

if mode == "Depth Estimation":
    depth_model = load_depth_model()
elif mode == "Object Detection (YOLOv8)":
    yolo_model = load_yolo_model()

FRAME_WINDOW = st.image([])

run_button = st.button("Start Webcam")

if run_button:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera Error. Could not retrieve frame.")
            break

        img = frame.copy()
        if mode == "Depth Estimation":
            out = run_depth(img, depth_model)
        else:
            out = run_yolo(img, yolo_model)

        # Convert BGR to RGB for Streamlit
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(out)

        # Exit button for breaking the loop (see workaround below)
        if st.button("Stop"):
            break
    cap.release()
    cv2.destroyAllWindows()
