import sys
sys.path.append("ultralytics/")
from ultralytics import YOLO

model = YOLO("yolo11n.yaml")
model = YOLO("yolo11n.pt")

results = model.train(data="../../cfg/bdd_custom.yaml", epochs=30, batch=8)
results = model.val()

success = model.export(format="onnx")
