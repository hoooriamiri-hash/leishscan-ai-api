"""LeishScan AI inference: Clinical lesion (DenseNet121) + Parasite present (YOLOv8n) -> System output."""
from pathlib import Path
import io, glob
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO

BASE = Path(__file__).parent
LESION_THRESHOLD = 0.25      # probability >= this => Clinical lesion = yes
YOLO_CONF = 0.25             # min detection confidence for amastigote
LESION_SIZE = 256
YOLO_IMGSZ = 1024
DEVICE = "cpu"

_tf = transforms.Compose([
    transforms.Resize((LESION_SIZE, LESION_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def _load_img(x):
    if isinstance(x, Image.Image): return x.convert("RGB")
    if isinstance(x, (bytes, bytearray)): return Image.open(io.BytesIO(x)).convert("RGB")
    return Image.open(x).convert("RGB")

def _build(sd):
    m = models.densenet121(weights=None)
    if "classifier.3.weight" in sd:
        m.classifier = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))
    else:
        m.classifier = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 1))
    return m

def _load_lesion_models():
    nets = []
    for f in sorted(glob.glob(str(BASE / "lesion" / "best_model_fold*.pt"))):
        ck = torch.load(f, map_location=DEVICE)
        sd = ck.get("model_state_dict", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        m = _build(sd)
        m.load_state_dict(sd, strict=True)
        nets.append(m.to(DEVICE).eval())
    return nets

_lesion_nets = _load_lesion_models()
_yolo = YOLO(str(BASE / "microscopy" / "best.pt"))

OUTCOMES = {
    (True, True):   ("Suspected CL, parasites demonstrated",
                     "Clinical assessment for initiation of CL treatment"),
    (True, False):  ("Suspected CL; CL not excluded. Parasites may not be demonstrable, particularly in chronic lesions",
                     "Skin biopsy for histopathological examination (HP) to further investigate CL"),
    (False, True):  ("Atypical presentation / clinical suspicion",
                     "Clinical assessment for initiation of CL treatment"),
    (False, False): ("CL is not the leading diagnosis, but CL needs to be excluded",
                     "Skin biopsy and further investigations to exclude CL and establish alternative diagnoses"),
}

@torch.no_grad()
def predict_lesion(img):
    x = _tf(_load_img(img)).unsqueeze(0).to(DEVICE)
    probs = [torch.sigmoid(n(x)).item() for n in _lesion_nets]
    return sum(probs) / len(probs)

def predict_parasite(img):
    res = _yolo.predict(_load_img(img), imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)[0]
    dets = []
    for b in res.boxes:
        name = res.names[int(b.cls)]
        dets.append({"class": name, "confidence": float(b.conf), "box_xyxy": [float(v) for v in b.xyxy[0]]})
    n_amast = sum(1 for d in dets if d["class"] == "amastigote")
    return n_amast, dets

def predict(lesion_image, microscopy_image):
    p = predict_lesion(lesion_image)
    n_amast, dets = predict_parasite(microscopy_image)
    lesion_yes, parasite_yes = p >= LESION_THRESHOLD, n_amast > 0
    output, next_step = OUTCOMES[(lesion_yes, parasite_yes)]
    return {
        "clinical_lesion": {"present": lesion_yes, "probability": round(p, 4), "threshold": LESION_THRESHOLD},
        "parasite_present": {"present": parasite_yes, "amastigote_count": n_amast, "detections": dets},
        "system_output": output,
        "next_step": next_step,
        "disclaimer": "Decision-support tool for research use. Not a substitute for clinical judgement.",
    }
