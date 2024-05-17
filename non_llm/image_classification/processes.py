import numpy as np
import utils
from skimage.transform import resize

def blur(x):
    x_pad = np.pad(x, [(0, 0), (0, 0), (1, 1), (1, 1)])
    x_pad = (x_pad[:, :, 1:] + x_pad[:, :, :-1])/2
    x_pad = (x_pad[:, :, :, 1:] + x_pad[:, :, :, :-1])/2
    return x_pad

def resize_image(image, size):
    resized_image = resize(image, size)
    return resized_image

def classify(x, model_1, model_2):
    orig_size = x.shape
    x = resize_image(x)
    x = model_1(x)
    x = blur(x)
    x = model_2(x)
    x = resize_image(x, orig_size)
    return utils.to_numpy(x)

def crop(input_1d, output_length=64):
    return input_1d[:output_length]
    