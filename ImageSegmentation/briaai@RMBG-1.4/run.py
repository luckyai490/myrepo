import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
# import gradio as gr
# from gradio_imageslider import ImageSlider
# from briarmbg import BriaRMBG
from transformers import AutoModelForImageSegmentation
import PIL
from PIL import Image
from typing import Tuple


# net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
net = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
net.eval()    

    
def resize_image(image):
    image = image.convert('RGB')
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image


def process(image):

    # prepare input
    # orig_image = Image.fromarray(image)
    orig_image = Image.open(image)
    w,h = orig_im_size = orig_image.size
    image = resize_image(orig_image)
    im_np = np.array(image)
    im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2,0,1)
    im_tensor = torch.unsqueeze(im_tensor,0)
    im_tensor = torch.divide(im_tensor,255.0)
    im_tensor = normalize(im_tensor,[0.5,0.5,0.5],[1.0,1.0,1.0])
    if torch.cuda.is_available():
        im_tensor=im_tensor.cuda()

    #inference
    result=net(im_tensor)
    # post process
    result = torch.squeeze(F.interpolate(result[0][0], size=(h,w), mode='bilinear') ,0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result-mi)/(ma-mi)    
    # image to pil
    result_array = (result*255).cpu().data.numpy().astype(np.uint8)
    pil_mask = Image.fromarray(np.squeeze(result_array))
    # add the mask on the original image as alpha channel
    new_im = orig_image.copy()
    new_im.putalpha(pil_mask)
    return new_im
    # return [new_orig_image, new_im]
import sys
examples = sys.argv[1] if len(sys.argv) > 1 else '../../images/1.jpg'
# examples = [['../../images/1.jpg']]
image = process(examples)
image.save(sys.argv[2] if len(sys.argv) > 2 else 'result.png')

