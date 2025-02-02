from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageEnhance


import torch

url = "../../images/2.jpg"
image = Image.open(url)

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]


    draw = ImageDraw.Draw(image)
    draw.rectangle([(box[0], box[1]), (box[2], box[3])], fill=None, outline="red", width=2)
    draw.text((box[0], box[1]), "{}: {}".format(model.config.id2label[label.item()], round(score.item(), 2)), fill="blue")

image.save("out.jpg")
image.show()