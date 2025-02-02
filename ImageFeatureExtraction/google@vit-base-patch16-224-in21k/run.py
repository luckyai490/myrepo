from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import sys

url = sys.argv[1] if len(sys.argv) > 1 else '../../images/1.jpg'
image = Image.open(url)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)