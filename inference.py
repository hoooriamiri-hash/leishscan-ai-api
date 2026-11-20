"""LeishScan AI inference (pure ONNX Runtime, no torch/ultralytics).

Clinical lesion (5-fold DenseNet121 ONNX ensemble) + Giemsa microscopy
(YOLOv8n `microscopy_best.onnx`, 1-5 fields aggregated) -> System output.
"""
import io, glob, ast
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image

BASE = Path(__file__).parent
MODEL_DIR = BASE / "onnx_models"

LESION_THRESHOLD = 0.25      # probability >= this => Clinical lesion = yes
LESION_SIZE = 256
LESION_OUTPUT_IS_LOGIT = True  # set False if your lesion ONNX files already end in a sigmoid
YOLO_CONF = 0.25             # min detection confidence for amastigote
YOLO_IOU = 0.7               # NMS IoU threshold (ultralytics default)
YOLO_IMGSZ = 1024            # used only if the ONNX input size is dynamic
MAX_FIELDS = 5
MAX_DETS_PER_FIELD = 40      # keeps the GitHub issue body small; counts are always exact

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_PROVIDERS = ["CPUExecutionProvider"]


def _load_img(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x)).convert("RGB")
    return Image.open(x).convert("RGB")


# ---------------- model loading ----------------
def _load_lesion_sessions():
    files = [f for f in sorted(glob.glob(str(MODEL_DIR / "*.onnx")))
             if "microscopy" not in Path(f).name.lower()]
    if not files:
        raise FileNotFoundError("No lesion ONNX models found in onnx_models/")
    return [ort.InferenceSession(f, providers=_PROVIDERS) for f in files]


def _yolo_names(sess):
    try:
        raw = sess.get_modelmeta().custom_metadata_map.get("names")
        return {int(k): str(v) for k, v in ast.literal_eval(raw).items()}
    except Exception:
        return {}


_lesion_sessions = _load_lesion_sessions()
_yolo = ort.InferenceSession(str(MODEL_DIR / "microscopy_best.onnx"), providers=_PROVIDERS)
_yolo_in = _yolo.get_inputs()[0]
_yolo_size = _yolo_in.shape[2] if isinstance(_yolo_in.shape[2], int) else YOLO_IMGSZ
_yolo_names_map = _yolo_names(_yolo) or {0: "amastigote"}

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


# ---------------- lesion (macroscopic) ----------------
def _prep_lesion(img):
    a = np.asarray(img.resize((LESION_SIZE, LESION_SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    a = (a - _MEAN) / _STD
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None], dtype=np.float32)


def predict_lesion(img):
    x = _prep_lesion(_load_img(img))
    probs = []
    for s in _lesion_sessions:
        v = float(np.asarray(s.run(None, {s.get_inputs()[0].name: x})[0]).reshape(-1)[0])
        probs.append(1.0 / (1.0 + np.exp(-v)) if LESION_OUTPUT_IS_LOGIT else v)
    return sum(probs) / len(probs)


# ---------------- microscopy (YOLOv8n ONNX) ----------------
def _letterbox(img, size):
    w, h = img.size
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    a = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None]), r, px, py


def _nms(boxes, scores, iou_thr):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1, yy1 = np.maximum(x1[i], x1[rest]), np.maximum(y1[i], y1[rest])
        xx2, yy2 = np.minimum(x2[i], x2[rest]), np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def _detect_field(img):
    img = _load_img(img)
    W, H = img.size
    x, r, px, py = _letterbox(img, _yolo_size)
    out = _yolo.run(None, {_yolo_in.name: x})[0]
    p = out[0] if out.ndim == 3 else out
    if p.shape[0] < p.shape[1]:          # (4+nc, N) -> (N, 4+nc)
        p = p.T
    scores = p[:, 4:]
    cls = scores.argmax(1)
    conf = scores.max(1)
    m = conf >= YOLO_CONF
    p, cls, conf = p[m], cls[m], conf[m]
    if len(p) == 0:
        return []
    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    keep = _nms(xyxy + cls[:, None] * 4096.0, conf, YOLO_IOU)   # class-aware NMS
    dets = []
    for i in keep:
        x1 = float(np.clip((xyxy[i, 0] - px) / r, 0, W)); y1 = float(np.clip((xyxy[i, 1] - py) / r, 0, H))
        x2 = float(np.clip((xyxy[i, 2] - px) / r, 0, W)); y2 = float(np.clip((xyxy[i, 3] - py) / r, 0, H))
        dets.append({"class": _yolo_names_map.get(int(cls[i]), f"class_{int(cls[i])}"),
                     "confidence": round(float(conf[i]), 4),
                     "box_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]})
    return dets


def predict_parasite(images):
    """images: one image or a list of 1-5 Giemsa fields. Aggregates across all fields."""
    if not isinstance(images, (list, tuple)):
        images = [images]
    images = list(images)[:MAX_FIELDS]
    per_slide, all_dets = [], []
    for idx, im in enumerate(images, start=1):
        dets = _detect_field(im)
        n = sum(1 for d in dets if d["class"].lower() == "amastigote")
        per_slide.append(n)
        for d in sorted(dets, key=lambda d: -d["confidence"])[:MAX_DETS_PER_FIELD]:
            all_dets.append({"slide": idx, **d})
    return sum(per_slide), per_slide, all_dets


def predict(lesion_image, microscopy_images):
    p = predict_lesion(lesion_image)
    n_amast, per_slide, dets = predict_parasite(microscopy_images)
    lesion_yes, parasite_yes = p >= LESION_THRESHOLD, n_amast > 0
    output, next_step = OUTCOMES[(lesion_yes, parasite_yes)]
    return {
        "clinical_lesion": {"present": bool(lesion_yes), "probability": round(float(p), 4), "threshold": LESION_THRESHOLD},
        "parasite_present": {"present": bool(parasite_yes), "amastigote_count": int(n_amast),
                             "slides_analyzed": len(per_slide), "per_slide_counts": per_slide,
                             "detections": dets},
        "system_output": output,
        "next_step": next_step,
        "disclaimer": "Decision-support tool for research use. Not a substitute for clinical judgement.",
    }
