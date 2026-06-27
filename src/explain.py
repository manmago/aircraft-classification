"""Grad-CAM: visualise which image regions drove the model's prediction.

Grad-CAM = Gradient-weighted Class Activation Mapping. For a chosen class, it
highlights the image regions that most increased that class's score — the
image-world version of 'feature importance'.

    from src.explain import grad_cam
    heatmap, class_idx = grad_cam(model, 'plane.jpg', device='cuda')
"""
import torch
import torch.nn.functional as F
from PIL import Image

from src.predict import infer_tf


def grad_cam(model, image, class_idx=None, device='cpu', target_layer=None, tf=None):
    """Compute a Grad-CAM heatmap.

    Args:
        model: the trained classifier.
        image: a PIL image or path.
        class_idx: which class to explain. None -> the model's predicted class.
        target_layer: conv layer to read. None -> ResNet-18's last conv block.

    Returns (heatmap, class_idx): heatmap is a 224x224 numpy array in [0, 1].
    """
    if target_layer is None:
        target_layer = model.layer4[-1]   # last conv block of ResNet-18 (7x7 grid)

    if isinstance(image, str):
        image = Image.open(image)
    x = (tf or infer_tf)(image.convert('RGB')).unsqueeze(0).to(device)

    # Hooks capture the layer's activations (forward) and gradients (backward).
    store = {}
    h1 = target_layer.register_forward_hook(
        lambda m, i, o: store.update(acts=o.detach()))
    h2 = target_layer.register_full_backward_hook(
        lambda m, gi, go: store.update(grads=go[0].detach()))

    model.eval()
    model.zero_grad()
    logits = model(x)
    if class_idx is None:
        class_idx = logits.argmax(dim=1).item()
    logits[0, class_idx].backward()       # gradients flow back to target_layer

    h1.remove()
    h2.remove()

    acts = store['acts'][0]               # [C, h, w]  what the layer detected
    grads = store['grads'][0]             # [C, h, w]  how much each mattered
    weights = grads.mean(dim=(1, 2))      # [C]  importance of each feature map
    cam = F.relu((weights[:, None, None] * acts).sum(dim=0))  # [h, w] weighted sum

    # Upsample the coarse 7x7 map to the 224x224 input and normalise to [0, 1].
    cam = F.interpolate(cam[None, None], size=(224, 224),
                        mode='bilinear', align_corners=False)[0, 0]
    cam = cam.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, class_idx
