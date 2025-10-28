# modules/gpt_analysis.py
import numpy as np
from PIL import Image
import io

def analyze_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)
    mean_val = np.mean(arr)
    vegetation = (mean_val / 255.0)
    if vegetation > 0.5:
        return "High vegetation density detected"
    else:
        return "Low vegetation or urban area"