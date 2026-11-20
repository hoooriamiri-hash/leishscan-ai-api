import glob
import inspect
from pathlib import Path
import onnx
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
lesion_files = sorted(glob.glob(str(BASE / "lesion" / "best_model_fold*.pt")))
if not lesion_files:
    raise FileNotFoundError(f"No lesion model checkpoints found in {BASE / 'lesion'}")

dummy_lesion = torch.randn(1, 3, 256, 256)
export_params = inspect.signature(torch.onnx.export).parameters

for f in lesion_files:
    ck = torch.load(f, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    model = build_densenet(sd)
    model.load_state_dict(sd, strict=True)
    model.eval()

    out_path = OUT / f"{Path(f).stem}.onnx"
    export_kwargs = {
        "input_names": ["input"],
        "output_names": ["logits"],
        "opset_version": 14,
        "do_constant_folding": True,
    }
    if "dynamo" in export_params:
        export_kwargs["dynamo"] = False

    torch.onnx.export(model, dummy_lesion, str(out_path), **export_kwargs)

    # Ensure all weights are embedded inside a single .onnx file for browser inference
    onnx_model = onnx.load(str(out_path), load_external_data=True)
    onnx.save_model(onnx_model, str(out_path), save_as_external_data=False)
    data_file = Path(f"{out_path}.data")
    if data_file.exists():
        data_file.unlink()

    print(f"Exported {out_path.name} ({out_path.stat().st_size / 1e6:.2f} MB)")


# 2. Export YOLOv8n microscopy detector to ONNX (imgsz=1024)
yolo_path = BASE / "microscopy" / "best.pt"
if not yolo_path.exists():
    raise FileNotFoundError(f"Microscopy checkpoint not found at {yolo_path}")

yolo = YOLO(str(yolo_path))
exported = yolo.export(format="onnx", imgsz=1024, opset=14, simplify=True)
dest_yolo = OUT / "microscopy_best.onnx"
Path(exported).replace(dest_yolo)
print(f"Exported {dest_yolo.name} ({dest_yolo.stat().st_size / 1e6:.2f} MB)")
