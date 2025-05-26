import torch
from ultralytics import YOLO
import easyocr
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration


@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    yolo_model = YOLO("yolo11n-seg.pt").to("cpu")

    ocr_reader = easyocr.Reader(['en'], gpu=False, download_enabled=False)

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model.to(device)

    return yolo_model, ocr_reader, processor, caption_model     





