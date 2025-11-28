# 04_train_teacher_PCB_i7-6600U_CPU_ONLY.py
# OPTIMIZED FOR VERY WEAK CPU → WILL ACTUALLY FINISH!
from ultralytics import YOLO
import torch

print("=" * 80)
print("PCB DEFECT DETECTION - TEACHER MODEL")
print("Optimized for Intel i7-6600U (2C/4T) - NO GPU")
print("Your CPU is old → we go FULL survival mode but still get good mAP!")
print("=" * 80)

# Force CPU
device = 'cpu'
print(f"Using device: CPU ({torch.get_num_threads()} threads)")

# Load a MUCH lighter model → YOLOv8n or YOLOv8s is the only realistic choice
# YOLOv8x on your CPU = 30+ days → impossible
model = YOLO('yolov8n.pt')        # ← nano = 3.2M params (perfect)
# model = YOLO('yolov8s.pt')      # ← small = 11M params (still okay, ~2.5x slower)

data_yaml = r"E:\IBA_MS_DS 2026\Computer Vision\Project\PCB_DATASET\yolo_dataset\data.yaml"

results = model.train(
    data=data_yaml,
    epochs=100,                  # 100 is enough with good augmentations
    imgsz=640,                   # 640 max! 1024 = 4–5x slower → not worth it on your CPU
    batch=4,                     # 4 is the maximum safe on 2-core CPU
    device=device,
    
    # CRITICAL FOR YOUR CPU
    workers=4,                   # Exactly your thread count → 100% CPU usage, no hanging
    cache='disk',                # RAM is probably 8–16GB → disk cache is safer
    pretrained=True,
    
    # Optimizer (AdamW is best)
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    
    # Small object focus (Spur class)
    box=7.5,
    cls=0.5,
    dfl=1.5,
    
    # Reduced but still strong augmentations (heavy ones kill your CPU)
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=5.0,
    translate=0.1,
    scale=0.5,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,                  # Keep mosaic → great for small defects
    mixup=0.15,                  # Reduced → was too slow
    copy_paste=0.0,              # Too heavy → disable
    close_mosaic=10,
    
    # Speed & logging
    patience=20,
    save_period=10,
    project='runs/teacher',
    name='PCB_teacher_i7-6600U_yolov8n_640px',
    exist_ok=True,
    plots=True,
    amp=True,                    # Helps a tiny bit even on CPU
    verbose=True
)

print("\nTRAINING COMPLETE!")
print("Now validating on test set...")

# Final evaluation
val_results = model.val(data=data_yaml, split='test', imgsz=640, batch=4)

print(f"\nFINAL RESULTS (yolov8n @ 640px):")
print(f"   mAP@50     : {val_results.box.map50:.4f}")
print(f"   mAP@50-95  : {val_results.box.map:.4f}")

# Export for deployment
model.export(format='onnx', imgsz=640)
print("ONNX exported!")

print("\nBest model saved to:")
print("   runs/teacher/PCB_teacher_i7-6600U_yolov8n_640px/weights/best.pt")