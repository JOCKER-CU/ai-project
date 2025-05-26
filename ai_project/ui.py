import streamlit as st
from PIL import Image
import numpy 
import tempfile
import cv2
import os
import ocr  # Assuming this is your OCR module
import model

from bg_remove import remove_background  # Assuming this is your background removal function
import os
import streamlit as st
import time

# import os
# os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
# Initialize OCR reader (only once)
# @st.cache_resource
# def init_ocr_reader():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     return easyocr.Reader(['en'], gpu=device.type == 'cuda')



# Background Removal Functions

# Main App
def main():
    st.title("Dual-Model Image Processing App")

 # Model loading section - runs only once per session
    if 'models' not in st.session_state:
        # First-time loading visual flow
        if 'first_load' not in st.session_state:
            loading_placeholder = st.empty()
            
            with loading_placeholder.container():
                with st.spinner("🚀 Loading AI models for the first time..."):
                    st.session_state.yolo, st.session_state.ocr = model.load_models()
                    st.success("✅ YOLO model loaded")
                    st.success("✅ EasyOCR initialized")
                
                time.sleep(2)  # Show messages for 2 seconds
            
            loading_placeholder.empty()
            st.session_state.first_load = False  # Mark first load complete
            st.session_state.models_loaded = True
            
            # Small persistent indicator
            st.toast("Models ready!", icon="🤖")
        
        # Subsequent loads will skip the visual flow
        else:
            st.session_state.yolo, st.session_state.ocr = model.load_models()
            st.session_state.models_loaded = True
    
    # Only show the rest of the UI after models are loaded
    if not st.session_state.get('models_loaded', False):
        st.warning("Please wait while models are loading...")
        st.stop()  # This halts execution until models are ready
    
    # Now show the main UI
    st.write("Upload an image and choose processing method")
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)

        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name

        # Processing options
        option = st.radio("Select processing method:", 
                         ["OCR (Text Extraction)", "Object Segmentation"])

        if st.button("Process Image"):
            if option == "OCR (Text Extraction)":
                with st.spinner("Extracting text..."):
                    image_rgb, extracted_text, boxes = ocr.process_image_for_ocr(temp_path, st.session_state.ocr)
                    
                    if extracted_text:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Extracted Text")
                            for text in extracted_text:
                                st.write(text)
                        with col2:
                            st.subheader("Text Detection")
                            annotated_image = ocr.draw_bounding_boxes(image_rgb, boxes)
                            st.image(annotated_image, caption="Detected Text", use_container_width=True)
                    else:
                        st.warning("No text detected in the image")

            elif option == "Object Segmentation":
                with st.spinner("Removing background..."):
                    segmented_img, object_names, original_rgb = remove_background(temp_path, st.session_state.yolo)
                    if segmented_img is not None:
                        # Display results in two columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Detected Objects")
                            if object_names:
                                for obj in object_names:
                                    st.write(f"- {obj}")
                            else:
                                st.write("No objects detected")
                        
                        with col2:
                            st.subheader("Segmented Image")
                            # Convert to PIL Image for display
                            pil_image = Image.fromarray(cv2.cvtColor(segmented_img, cv2.COLOR_BGRA2RGBA))
                            st.image(pil_image, caption="Background Removed", use_container_width=True)
                    else:
                        st.warning("Segmentation failed")

        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass

if __name__ == "__main__":
    main()
