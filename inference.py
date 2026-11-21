"""LeishScan AI inference (pure ONNX Runtime, no torch/ultralytics).

Clinical lesion (5-fold DenseNet121 ONNX ensemble) + Giemsa microscopy
(YOLOv8n `microscopy_best.onnx`, 1-5 slides aggregated) -> System output.

YOLO pre/post-processing mirrors Ultralytics `model.predict(imgsz=1024, conf, iou)`:
RGB, /255, centred letterbox (pad 114, cv2 INTER_LINEAR), class-aware NMS,
boxes mapped back to the original image.

Local test (same code path as GitHub Actions):
    python inference.py lesion.jpg slide1.jpg slide2.jpg slide3.jpg
"""
import io, glob, ast, json, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

try:
    import cv2  # optional: gives bit-identical resizing to Ultralytics
except Exception:  # pragma: no cover
    cv2 = None

BASE = Path(__file__).parent
MODEL_DIR = BASE / "onnx_models"

LESION_THRESHOLD = 0.25        # probability >= this => Clinical lesion = yes
LESION_SIZE = 256
LESION_OUTPUT_IS_LOGIT = True  # set False if your lesion ONNX files already end in a sigmoid
YOLO_CONF = 0.25               # min detection confidence
YOLO_IOU = 0.45                # NMS IoU (Ultralytics predict() default is 0.7 - change to match your local run)
YOLO_IMGSZ = 1024              # used only if the ONNX input size is dynamic
MAX_FIELDS = 5
MAX_DETS_PER_FIELD = 40        # keeps the GitHub issue body small; counts are always exact

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_PROVIDERS = ["CPUExecutionProvider"]
_S = {}

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


def _load_img(x, exif=False):
    if isinstance(x, Image.Image):
        im = x
    elif isinstance(x, (bytes, bytearray)):
        im = Image.open(io.BytesIO(x))
    else:
        im = Image.open(x)
    if exif:                      # cv2.imread (used by Ultralytics) honours EXIF orientation
        im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def _models():
    """Lazy-load all ONNX sessions once."""
    if _S:
        return _S
    lesion_files = [f for f in sorted(glob.glob(str(MODEL_DIR / "*.onnx")))
                    if "microscopy" not in Path(f).name.lower()]
    if not lesion_files:
        raise FileNotFoundError("No lesion ONNX models found in onnx_models/")
    yolo = ort.InferenceSession(str(MODEL_DIR / "microscopy_best.onnx"), providers=_PROVIDERS)
    yin = yolo.get_inputs()[0]
    try:
        names = {int(k): str(v) for k, v in
                 ast.literal_eval(yolo.get_modelmeta().custom_metadata_map.get("names")).items()}
    except Exception:
        names = {}
    _S.update(
        lesion=[ort.InferenceSession(f, providers=_PROVIDERS) for f in lesion_files],
        yolo=yolo, yin=yin,
        size=yin.shape[2] if isinstance(yin.shape[2], int) else YOLO_IMGSZ,
        fp16="float16" in yin.type, names=names,
    )
    print(f"[init] lesion folds={len(lesion_files)} yolo_input={yin.shape} type={yin.type} "
          f"names={names or 'NOT IN METADATA'} cv2={'yes' if cv2 else 'no (PIL fallback)'}", flush=True)
    return _S


# ---------------- lesion (macroscopic) ----------------
def _prep_lesion(img):
    a = np.asarray(img.resize((LESION_SIZE, LESION_SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    a = (a - _MEAN) / _STD
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None], dtype=np.float32)


def predict_lesion(img):
    x = _prep_lesion(_load_img(img))
    probs = []
    for s in _models()["lesion"]:
        v = float(np.asarray(s.run(None, {s.get_inputs()[0].name: x})[0]).reshape(-1)[0])
        probs.append(1.0 / (1.0 + np.exp(-v)) if LESION_OUTPUT_IS_LOGIT else v)
    return sum(probs) / len(probs)


# ---------------- microscopy (YOLOv8n ONNX) ----------------
def _letterbox(arr, size):
    """Ultralytics LetterBox(auto=False, center=True): returns CHW float batch, ratio, left, top."""
    h, w = arr.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) / 2, (size - nh) / 2
    if (w, h) != (nw, nh):
        arr = (cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_LINEAR) if cv2
               else np.asarray(Image.fromarray(arr).resize((nw, nh), Image.BILINEAR)))
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = arr
    x = canvas.astype(np.float32) / 255.0                       # RGB, 0-1
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), r, left, top


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
        order = rest[inter / (areas[i] + areas[rest] - inter + 1e-9) <= iou_thr]
    return keep


def _is_target(cid, nc, names):
    """Does this class id count as an amastigote?"""
    if names:
        n = names.get(cid, "").lower()
        return len(names) == 1 or "amastigote" in n or "leish" in n
    return nc == 1 or cid == 0   # no metadata: single-class model, else assume class 0


def _detect_field(src):
    S = _models()
    img = _load_img(src, exif=True)
    W, H = img.size
    x, r, left, top = _letterbox(np.asarray(img), S["size"])
    if S["fp16"]:
        x = x.astype(np.float16)
    out = np.asarray(S["yolo"].run(None, {S["yin"].name: x})[0], dtype=np.float32)
    p = out[0] if out.ndim == 3 else out

    if p.shape[1] == 6 and p.shape[0] > 6:       # end-to-end export (nms=True): [x1,y1,x2,y2,conf,cls]
        m = p[:, 4] >= YOLO_CONF
        xyxy, conf, cls = p[m, :4], p[m, 4], p[m, 5].astype(int)
        max_conf, nc = float(p[:, 4].max()) if len(p) else 0.0, int(max(cls.max() + 1, 1)) if len(cls) else 1
    else:                                        # raw export: (4+nc, N) -> (N, 4+nc)
        if p.shape[0] < p.shape[1]:
            p = p.T
        scores = p[:, 4:]
        nc = scores.shape[1]
        cls_all, conf_all = scores.argmax(1), scores.max(1)
        max_conf = float(conf_all.max()) if len(conf_all) else 0.0
        m = conf_all >= YOLO_CONF
        q, cls, conf = p[m], cls_all[m], conf_all[m]
        if len(q) == 0:
            return [], 0, max_conf
        cx, cy, w, h = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        keep = _nms(xyxy + cls[:, None] * 4096.0, conf, YOLO_IOU)   # class-aware NMS
        xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]

    xyxy = (xyxy - np.array([left, top, left, top], dtype=np.float32)) / r      # back to original pixels
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, W)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, H)
    dets, n_target = [], 0
    for b, c, k in zip(xyxy, conf, cls):
        k = int(k)
        tgt = _is_target(k, nc, S["names"])
        n_target += int(tgt)
        dets.append({"class": S["names"].get(k, "amastigote" if tgt else f"class_{k}"),
                     "confidence": round(float(c), 4),
                     "box_xyxy": [round(float(v), 1) for v in b]})
    return dets, n_target, max_conf


def predict_parasite(images):
    """images: one image or a list of 1-5 Giemsa slides (paths / bytes / PIL). Aggregates across all slides."""
    if not isinstance(images, (list, tuple)):
        images = [images]
    per_slide, max_conf, all_dets = [], [], []
    for idx, im in enumerate(list(images)[:MAX_FIELDS], start=1):
        dets, n, mc = _detect_field(im)
        per_slide.append(n)
        max_conf.append(round(mc, 4))
        print(f"[slide {idx}] detections={len(dets)} amastigotes={n} max_conf={mc:.4f}", flush=True)
        for d in sorted(dets, key=lambda d: -d["confidence"])[:MAX_DETS_PER_FIELD]:
            all_dets.append({"slide": idx, **d})
    return sum(per_slide), per_slide, max_conf, all_dets


def predict(lesion_image, microscopy_images):
    p = predict_lesion(lesion_image)
    n_amast, per_slide, max_conf, dets = predict_parasite(microscopy_images)
    lesion_yes, parasite_yes = p >= LESION_THRESHOLD, n_amast > 0
    output, next_step = OUTCOMES[(lesion_yes, parasite_yes)]
    return {
        "clinical_lesion": {"present": bool(lesion_yes), "probability": round(float(p), 4), "threshold": LESION_THRESHOLD},
        "parasite_present": {"present": bool(parasite_yes), "amastigote_count": int(n_amast),
                             "slides_analyzed": len(per_slide), "per_slide_counts": per_slide,
                             "per_slide_max_conf": max_conf, "detections": dets},
        "system_output": output,
        "next_step": next_step,
        "disclaimer": "Decision-support tool for research use. Not a substitute for clinical judgement.",
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python inference.py lesion.jpg slide1.jpg [slide2.jpg ... slide5.jpg]")
    print(json.dumps(predict(sys.argv[1], sys.argv[2:7]), indent=2))
