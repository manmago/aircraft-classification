"""Inference utilities: load the trained model and predict on a single image.

    from src.predict import load_model, predict
    model, classes = load_model('../models/resnet18_aircraft5.pt')
    print(predict(model, classes, 'my_plane.jpg'))
"""
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

from src.data import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

# Same as the eval transform in data.py, but WITHOUT remove_banner — real-world
# photos don't have the FGVC copyright banner, so we must not crop their bottoms.
infer_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_model(ckpt_path, device='cpu'):
    """Rebuild the architecture, load the saved weights, return (model, classes)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt['classes']

    model = resnet18()  # no ImageNet download — we load OUR weights below
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt['state_dict'])

    model.eval().to(device)
    return model, classes


@torch.no_grad()
def predict(model, classes, image, device='cpu'):
    """Predict the aircraft for a PIL image or an image path.

    Returns a list of (class_name, probability) sorted from most to least likely.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    image = image.convert('RGB')

    x = infer_tf(image).unsqueeze(0).to(device)   # add batch dim -> [1, 3, 224, 224]
    logits = model(x)                             # raw scores
    probs = logits.softmax(dim=1)[0].cpu()        # -> probabilities summing to 1

    return sorted(zip(classes, probs.tolist()), key=lambda kv: -kv[1])
