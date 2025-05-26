import easyocr
import cv2
import numpy as np
import torch

# def initialize_ocr_reader():
#     """Initialize the EasyOCR reader with GPU if available"""
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     return easyocr.Reader(['en'], gpu=device.type == 'cuda')



def process_image_for_ocr(image_path, reader):
    """Process image with EasyOCR"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image from path")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = reader.readtext(image_rgb)
        
        extracted_text = []
        boxes = []
        for (bbox, text, prob) in results:
            extracted_text.append(f"{text} (Confidence: {prob:.2f})")
            boxes.append(np.array(bbox, dtype=np.int32))
            
        return image_rgb, extracted_text, boxes
        
    except Exception as e:
        print(f"OCR error: {str(e)}")
        return None, [], []

def draw_bounding_boxes(image, boxes):
    """Draw bounding boxes on the image"""
    annotated_image = image.copy()
    for box in boxes:
        cv2.polylines(annotated_image, [box], True, (0, 255, 0), 2)
    return annotated_image
