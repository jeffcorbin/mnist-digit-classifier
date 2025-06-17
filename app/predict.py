import torch
import torchvision.transforms as transforms
from PIL import Image
from app.model import load_model
import datetime
import csv
import io
import base64
import numpy

model = load_model()

def predict_image(image_file):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_file).convert("L")
    image = transform(image).unsqueeze(0)  # Add batch dim
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0, predicted.item()].item()

    log_inference(predicted.item(), confidence)
    return predicted.item(), round(confidence * 100, 2)

def predict_base64_image(b64_string):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image_data = base64.b64decode(b64_string.split(",")[1])
    image = Image.open(io.BytesIO(image_data)).convert("L")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0, predicted.item()].item()

    log_inference(predicted.item(), confidence)
    return predicted.item(), round(confidence * 100, 2)

def log_inference(prediction, confidence):
    with open("inference_logs.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.datetime.now(), prediction, confidence])