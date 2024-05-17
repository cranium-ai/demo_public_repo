# Image Classification

This directory contains scripts for an image classification task using two models.

## Files

- `processes.py`: This script contains functions for image processing and classification. It includes functions to resize and blur images, and to classify images using two models.

- `model.py`: This script contains the implementation of an `autoencoder` model that denoises the input image and a `convolutional neural network` model that classifies its input image.

- `inference_image.py`: This script instantiates the model and applies relevant pre- and postprocessing functions.

## Usage

To classify an image, run:

`python inference_image.py <path_to_image>`