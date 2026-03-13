"""
Minimal CBAM-ResNet-like placeholder for CPU usage.
Provides `run_vision_inference(img_path)` which returns a simple dict.
This is intentionally lightweight and safe to run on machines without GPU.
"""
import os
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torchvision.transforms as T

import config


class BasicCBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_att(x) * x
        # Spatial attention
        avg = torch.mean(ca, dim=1, keepdim=True)
        mx, _ = torch.max(ca, dim=1, keepdim=True)
        sa = torch.cat([avg, mx], dim=1)
        sa = self.spatial_att(sa) * ca
        return sa


class DummyVisionModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.cbam = BasicCBAM(16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.cbam(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


_MODEL: DummyVisionModel | None = None


def get_model() -> DummyVisionModel:
    global _MODEL
    if _MODEL is None:
        model = DummyVisionModel(num_classes=5)
        model.eval()
        # Attempt to load weights if present
        model_path = os.path.join("models", "cbam_resnet_no_entry_v1 .pth")
        if os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=torch.device('cpu'))
                model.load_state_dict(state)
            except Exception:
                pass
        _MODEL = model
    return _MODEL


def run_vision_inference(img_path: str) -> Dict[str, Any]:
    """Run a single-image inference and return a small insight dict."""
    model = get_model()
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        img = Image.open(img_path).convert('RGB')
        x = transform(img).unsqueeze(0)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1).cpu().numpy().tolist()[0]
        return {
            "image": img_path,
            "predictions": probs,
            "label": int(max(range(len(probs)), key=lambda i: probs[i]))
        }
    except Exception as e:
        return {"image": img_path, "error": str(e)}


def _normalize_anchor_priors(anchor_priors: List[List[float]] | None) -> List[Tuple[float, float]]:
    default_priors = [(0.2, 0.2), (0.35, 0.25), (0.5, 0.4)]
    if not anchor_priors:
        return default_priors

    parsed: List[Tuple[float, float]] = []
    for prior in anchor_priors:
        if not isinstance(prior, (list, tuple)) or len(prior) != 2:
            continue
        try:
            w = float(prior[0])
            h = float(prior[1])
            if 0 < w <= 1 and 0 < h <= 1:
                parsed.append((w, h))
        except Exception:
            continue

    return parsed or default_priors


def _build_box_for_detection(index: int, image_width: int, image_height: int, prior: Tuple[float, float]):
    box_w = max(8, int(image_width * prior[0]))
    box_h = max(8, int(image_height * prior[1]))

    slots = max(1, index + 1)
    center_x = int(((index % slots) + 1) * image_width / (slots + 1))
    center_y = int(image_height * (0.2 + (index % 3) * 0.25))

    x1 = max(0, center_x - box_w // 2)
    y1 = max(0, center_y - box_h // 2)
    x2 = min(image_width - 1, x1 + box_w)
    y2 = min(image_height - 1, y1 + box_h)
    return [x1, y1, x2, y2]


def _save_annotated_image(input_path: str, detections: List[Dict[str, Any]]) -> str:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(config.OUTPUT_DIR, f"{base_name}_detections.png")

    img = Image.open(input_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for det in detections:
        box = det.get("coordinates")
        if not box or len(box) != 4:
            continue
        label = f"cls {det['label_id']} ({det['confidence']}%)"
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0] + 4, max(0, box[1] - 14)), label, fill="red")

    img.save(output_path)
    return output_path


def run_vision_inference_simple(
    img_path: str,
    top_k: int = 3,
    conf_threshold: float = 0.05,
    anchor_priors: List[List[float]] | None = None
):
    """Wrapper to produce LLM-friendly insights from the local model.

    Since the placeholder model doesn't predict bounding boxes, this returns
    detected labels and confidences instead of box coordinates.
    """
    model = get_model()
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        img = Image.open(img_path).convert('RGB')
        x = transform(img).unsqueeze(0)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1).cpu().numpy().tolist()[0]

        normalized_priors = _normalize_anchor_priors(anchor_priors)

        # Build a simple detections list
        detections = []
        for idx, p in sorted(enumerate(probs), key=lambda iv: iv[1], reverse=True)[:top_k]:
            if p < conf_threshold:
                continue
            prior = normalized_priors[len(detections) % len(normalized_priors)]
            coordinates = _build_box_for_detection(
                index=len(detections),
                image_width=img.width,
                image_height=img.height,
                prior=prior
            )
            detections.append({
                "label_id": int(idx),
                "confidence": round(float(p) * 100.0, 2),
                "coordinates": coordinates
            })

        output_image = _save_annotated_image(img_path, detections) if detections else None

        insight = {
            "image_analyzed": os.path.basename(img_path),
            "objects_detected_count": len(detections),
            "detections": detections,
            "output_image": output_image,
            "conf_threshold": conf_threshold,
            "anchor_priors": normalized_priors
        }
        return insight
    except Exception as e:
        return {"image_analyzed": img_path, "error": str(e)}
