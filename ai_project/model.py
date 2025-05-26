import torch
from ultralytics import YOLO
import easyocr
import streamlit as st


@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models sequentially to better manage memory
    yolo_model = YOLO("yolo11n-seg.pt").to(device)
    
    # Initialize EasyOCR with same device
    ocr_reader = easyocr.Reader(
        ['en'], 
        # gpu=device.type == 'cuda',
        gpu=False,
        download_enabled=False  # Disable automatic download
    )       
    
    
    return yolo_model, ocr_reader


