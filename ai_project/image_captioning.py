from PIL import Image

def image_captionings(image_path, processor, model):
    """Generate image caption using BLIP model"""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_length=50,num_beams=5,early_stopping=True)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Captioning error: {str(e)}")
        return None
