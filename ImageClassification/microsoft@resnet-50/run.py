from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from PIL import Image
import sys

url = sys.argv[1] if len(sys.argv) > 1 else '../../images/1.jpg'
image = Image.open(url)

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])