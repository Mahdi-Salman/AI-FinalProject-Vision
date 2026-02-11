import cv2
import numpy as np

def resize_with_pad(image, target_size=(640, 640), color=(114, 114, 114)):
    shape = image.shape[:2]
    ratio = min(target_size[0] / shape[0], target_size[1] / shape[1])
    
    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
    dw, dh = target_size[1] - new_unpad[0], target_size[0] - new_unpad[1]
    
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, ratio, (dw, dh)

def normalize_image(image):
    return image.astype(np.float32) / 255.0