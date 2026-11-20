"""LeishScan AI inference: Fast ONNX Runtime (DenseNet121 5-fold + YOLOv8n)."""
from pathlib import Path
import io
import glob
import math
import numpy as np
import onnxruntime as ort
from PIL import Image

BASE = Path(__file__).parent
LESION_THRESHOLD = 0.25      # probability >= this => Clinical lesion = yes
YOLO_CONF = 0.25             # min detection confidence for amastigote
YOLO_IOU = 0.45              # NMS IoU threshold
LESION_SIZE = 256
YOLO_IMGSZ = 1024

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

_sess_opts = ort.SessionOptions()
_sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

def _load_lesion_sessions():
    sessions = []
    onnx_files = sorted(glob.glob(str(BASE / "onnx_models" / "best_model_fold*.onnx")))
    if not onnx_files:
        raise FileNotFoundError("No DenseNet ONNX models found in onnx_models/")
    for f in onnx_files:
        sessions.append(ort.InferenceSession(f, _sess_opts, providers=["CPUExecutionProvider"]))
    return sessions

_lesion_sessions = _load_lesion_sessions()
_yolo_session = ort.InferenceSession(
    str(BASE / "onnx_models" / "microscopy_best.onnx"),
    _sess_opts,
    providers=["CPUExecutionProvider"]
)

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

def _load_img(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x)).convert("RGB")
    return Image.open(x).convert("RGB")

def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-float(x)))

def predict_lesion(img):
    pil_img = _load_img(img).resize((LESION_SIZE, LESION_SIZE), Image.BILINEAR)
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    x = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
    x = (x - MEAN) / STD

    probs = []
    for sess in _lesion_sessions:
        input_name = sess.get_inputs()[0].name
        logit = sess.run(None, {input_name: x})[0].squeeze()
        probs.append(_sigmoid(logit))
    return sum(probs) / len(probs)

def _letterbox(pil_img, new_shape=YOLO_IMGSZ):
    """Resize and pad image while maintaining aspect ratio (matches Ultralytics YOLO)."""
    w, h = pil_img.size
    r = min(new_shape / w, new_shape / h)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape - new_unpad[0]) / 2.0
    dh = (new_shape - new_unpad[1]) / 2.0

    resized = pil_img.resize(new_unpad, Image.BILINEAR)
    canvas = Image.new("RGB", (new_shape, new_shape), (114, 114, 114))
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    canvas.paste(resized, (left, top))
    return canvas, r, (dw, dh)

def _nms(boxes, scores, iou_threshold):
    """Pure NumPy Non-Maximum Suppression."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def predict_parasite(img):
    pil_img = _load_img(img)
    orig_w, orig_h = pil_img.size
    boxed, ratio, (dw, dh) = _letterbox(pil_img, YOLO_IMGSZ)

    arr = np.asarray(boxed, dtype=np.float32) / 255.0
    x = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]

    input_name = _yolo_session.get_inputs()[0].name
    raw_out = _yolo_session.run(None, {input_name: x})[0]

    preds = np.squeeze(raw_out, axis=0).T
    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    mask = confidences >= YOLO_CONF
    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes_xywh) == 0:
        return 0, []

    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0

    keep = _nms(boxes_xyxy, confidences, YOLO_IOU)

    class_names = {0: "amastigote"}
    dets = []
    for idx in keep:
        b = boxes_xyxy[idx]
        x1 = max(0.0, min(float(orig_w), (float(b[0]) - dw) / ratio))
        y1 = max(0.0, min(float(orig_h), (float(b[1]) - dh) / ratio))
        x2 = max(0.0, min(float(orig_w), (float(b[2]) - dw) / ratio))
        y2 = max(0.0, min(float(orig_h), (float(b[3]) - dh) / ratio))

        cls_id = int(class_ids[idx])
        name = class_names.get(cls_id, "amastigote")
        dets.append({
            "class": name,
            "confidence": float(confidences[idx]),
            "box_xyxy": [x1, y1, x2, y2]
        })

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
