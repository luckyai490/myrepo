from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch
import sys

url = sys.argv[1] if len(sys.argv) > 1 else "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(url)


processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

inputs = processor(image, return_tensors="pt")
outputs = model(**inputs)

image_width, image_height = image.size  

image_mask = outputs.mask[0]
image_indices = torch.nonzero(image_mask).squeeze()

image_scores = outputs.scores[0][image_indices]
image_keypoints = outputs.keypoints[0][image_indices]

keypoints = image_keypoints.detach().numpy()
scores = image_scores.detach().numpy()

valid_keypoints = [
    (kp, score) for kp, score in zip(keypoints, scores)
    if 0 <= kp[0] < image_width and 0 <= kp[1] < image_height
]

valid_keypoints, valid_scores = zip(*valid_keypoints)
valid_keypoints = torch.tensor(valid_keypoints)
valid_scores = torch.tensor(valid_scores)

print(valid_keypoints.shape)

plt.axis('off')
plt.imshow(image)
plt.scatter(
    valid_keypoints[:, 0], 
    valid_keypoints[:, 1], 
    s=valid_scores * 100, 
    c='red'
)
plt.show()