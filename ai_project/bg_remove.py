from ultralytics import YOLO
import cv2
import numpy as np
import torch
import os



# output_path = "C:\\Users\\hanna\\Downloads\\"
# image_path = "C:\\Users\\hanna\\Downloads\\dog.jpg"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = YOLO("yolo11n-seg.pt")  # load an official model
# print(f"Model loaded on device: {device}")
# @st.cache_resource
# def load_model():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = YourModelClass().to(device)  # Replace with your actual model
#     model.load_state_dict(torch.load("your_model_weights.pth"))
#     model.eval()
#     return model


# def remove_background(image_path, output_path):


#     image = cv2.imread(image_path)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = model(image_rgb, device=device, conf=0.5)[0]  # apply model and get the first result

#     if results.masks is None:
#         print("No mask detected")
#         exit(0)

#     print("Mask detected")
#     masks = results.masks.data
#     print(f"Number of masks: {masks.shape[0]}")

#     # Combine masks
#     combined_mask = torch.any(masks.bool(), dim=0).cpu().numpy()

#     # Resize mask if needed
#     if combined_mask.shape != image.shape[:2]:
#         combined_mask = cv2.resize(combined_mask.astype(np.uint8), (image.shape[1], image.shape[0])) * 255
#     else:
#         combined_mask = combined_mask.astype(np.uint8) * 255

#     # Create alpha channel
#     b, g, r = cv2.split(image)
#     alpha = combined_mask
#     rgba = cv2.merge((b, g, r, alpha))

#     # Get object names from results
#     object_names = set()
#     if hasattr(results, 'names'):  # If your model stores class names
#         for class_id in results.boxes.cls.int().tolist():
#             object_names.add(results.names[class_id])
#     else:
#         print("Warning: Object names not available from model.")

#     # Create filename using detected object names
#     object_str = "_".join(sorted(object_names)) if object_names else "detected"
#     filename = f"{output_path}{object_str}_masked.png"

#     # Save the image
#     cv2.imwrite(filename, rgba)
#     print(f"Saved as: {filename}")


# remove_background(image_path, output_path)

# bg_remove.py
import cv2
import numpy as np
import torch

def remove_background(image_path, model):
    """
    Removes background using YOLO segmentation
    Args:
        image_path: Path to input image
        model: Loaded YOLO model
    Returns:
        rgba_image: Processed RGBA image with transparency
        object_names: Set of detected object names
        original_image: Original RGB image
    """
    try:
        # Read and convert image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image from path")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply model
        device = next(model.model.parameters()).device  # Get model's device
        results = model(image_rgb, device=device, conf=0.5)[0]
        
        if results.masks is None:
            return None, set(), image_rgb
        
        # Process masks
        masks = results.masks.data
        combined_mask = torch.any(masks.bool(), dim=0).cpu().numpy()
        
    
 
        if results.masks is None:
            print("No mask detected")
            return None, set(), image_rgb

        print("Mask detected")
        masks = results.masks.data
        print(f"Number of masks: {masks.shape[0]}")

    # Combine masks
        combined_mask = torch.any(masks.bool(), dim=0).cpu().numpy()

        # Resize mask if needed
            
        if combined_mask.shape != image.shape[:2]:
            combined_mask = cv2.resize(combined_mask.astype(np.uint8), 
                                    (image.shape[1], image.shape[0])) * 255
        else:
            combined_mask = combined_mask.astype(np.uint8) * 255

        # Create alpha channel
        b, g, r = cv2.split(image)
        alpha = combined_mask
        rgba_image = cv2.merge((b, g, r, alpha))

        # Get object names from results
        object_names = set()
        if hasattr(results, 'names'):
            for class_id in results.boxes.cls.int().tolist():
                object_names.add(results.names[class_id])

        return rgba_image, object_names, image_rgb
    except Exception as e:
        print(f"Segmentation error: {str(e)}")
        return None, set(), None