import glob
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO

BASE = Path(__file__).parent
OUT = BASE / "onnx_models"
OUT.mkdir(exist_ok=True)

def build_densenet(sd):
    m = models.densenet121(weights=None)
    if "classifier.3.weight" in sd:
        m.classifier = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1)
        )
    else:
        m.classifier = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 1)
        )
    return m

# 1. Export all 5 DenseNet121 lesion folds to ONNX
dummy_lesion = torch.randn(1, 3, 256, 256)
for f in sorted(glob.glob(str(BASE / "lesion" / "best_model_fold*.pt"))):
    ck = torch.load(f, map_location="cpu")
    sd = ck.get("model_state_dict", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model = build_densenet(sd)
    model.load_state_dict(sd, strict=True)
    model.eval()
    out_path = OUT / f"{Path(f).stem}.onnx"
    torch.onnx.export(
        model,
        dummy_lesion,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"Exported {out_path.name}")

# 2. Export YOLOv8n microscopy detector to ONNX (imgsz=1024)
yolo = YOLO(str(BASE / "microscopy" / "best.pt"))
exported = yolo.export(format="onnx", imgsz=1024, opset=14, simplify=True)
Path(exported).replace(OUT / "microscopy_best.onnx")
print("Exported microscopy_best.onnx")
