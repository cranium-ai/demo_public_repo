import torch
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import argparse

# Load the model and processor from Hugging Face
model_name = "hustvl/yolos-tiny"
processor = YolosImageProcessor.from_pretrained(model_name)
model = YolosForObjectDetection.from_pretrained(model_name)

# Function to load an image
def load_image(image_path):
    image = Image.open(image_path)
    return image

# Function to perform object detection
def detect_objects(image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Process the outputs to extract bounding boxes and labels
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    return results

# Function to draw bounding boxes on the image
def draw_boxes(image, boxes, labels, scores):
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(boxes, labels, scores):
        draw.rectangle(box.tolist(), outline="red", width=3)
        draw.text((box[0], box[1]), f"{label} ({score:.2f})", fill="red")

    return image

# Main function
def main(image_path):
    # Load the image
    image = load_image(image_path)

    # Perform object detection
    results = detect_objects(image)

    # Draw bounding boxes
    boxes = results["boxes"]
    labels = results["labels"]
    scores = results["scores"]

    label_map = {i: label for i, label in enumerate(model.config.id2label.values())}
    label_names = [label_map[label.item()] for label in labels]

    image_with_boxes = draw_boxes(image, boxes, label_names, scores)

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image_with_boxes)
    plt.axis("off")
    plt.show()

# Replace 'path_to_your_image.jpg' with the path to your image file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify an image')
    parser.add_argument('image_path', help='Path to the image to be classified')
    args = parser.parse_args()
    
    main(args.image_path)
