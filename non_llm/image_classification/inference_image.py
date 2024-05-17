import numpy as np
import torch
import argparse
import cv2

from model import AllConvModelTorch_5, Autoencoder_5
from processes import classify

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify an image')
    parser.add_argument('image_path', help='Path to the image to be classified')
    args = parser.parse_args()

    MODEL_PATH = "../checkpoints/blur/final_checkpoint-1"
    model_1 = Autoencoder_5()
    model_2 = AllConvModelTorch_5(num_classes=10,
                            num_filters=64,
                            input_shape=[3, 32, 32])
    model_2.load_state_dict(
        torch.load((MODEL_PATH) + ".torchmodel"))
    
    image = cv2.imread(args.image_path)
    classify(image, model_1, model_2)