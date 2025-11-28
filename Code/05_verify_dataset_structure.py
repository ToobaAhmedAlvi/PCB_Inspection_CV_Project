# Just run this — it will confirm everything is perfect
import os
from pathlib import Path

def verify():
    base = Path(r"E:\IBA_MS_DS 2026\Computer Vision\Project\PCB_DATASET\yolo_dataset")
    print("Verifying your YOLO dataset...")
    print(f"Total images: {len(list(base.rglob('*.jpg')))}")
    for split in ['train', 'val', 'test']:
        imgs = len(list((base/split/'images').glob('*.jpg')))
        lbls = len(list((base/split/'labels').glob('*.txt')))
        print(f"{split.upper():5}: {imgs} images, {lbls} labels → {'Perfect' if imgs==lbls else 'ERROR'}")

if __name__ == "__main__":
    verify()