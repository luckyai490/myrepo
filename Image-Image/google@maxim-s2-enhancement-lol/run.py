from huggingface_hub import from_pretrained_keras
from PIL import Image

import tensorflow as tf
import numpy as np
import requests

url = "https://github.com/sayakpaul/maxim-tf/raw/main/images/Enhancement/input/748.png"
image = Image.open(requests.get(url, stream=True).raw)
image = np.array(image)
image = tf.convert_to_tensor(image)
image = tf.image.resize(image, (256, 256))

model = from_pretrained_keras("google/maxim-s2-enhancement-lol")
predictions = model.predict(tf.expand_dims(image, 0))