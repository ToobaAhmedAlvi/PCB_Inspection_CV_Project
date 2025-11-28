# resize_and_zip_FINAL_WORKING.py → RUN THIS EXACTLY
import os
from PIL import Image
import zipfile
import shutil
from pathlib import Path  # ← THIS WAS MISSING!

# CHANGE THESE TWO PATHS ONLY
SOURCE_FOLDER = r"E:\IBA_MS_DS 2026\Computer Vision\Project\PCB_DATASET\yolo_dataset"
OUTPUT_ZIP = r"E:\IBA_MS_DS 2026\Computer Vision\Project\PCB_DATASET\yolo_dataset_1024.zip"
TARGET_SIZE = 1024  # ← 1024 = best quality/size | Change to 640 for smaller (~8GB)

# Create temp folder
TEMP_DIR = "temp_resized_dataset"
os.makedirs(TEMP_DIR, exist_ok=True)

def resize_image(img_path, out_path):
    with Image.open(img_path) as img:
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        img.save(out_path, "JPEG", optimize=True, quality=92)

# Count total images
total_images = len(list(Path(SOURCE_FOLDER).rglob("*.[jp][pn]g")))
print(f"Found {total_images} images → Resizing to {TARGET_SIZE}×{TARGET_SIZE}...")

count = 0
with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
    for file_path in Path(SOURCE_FOLDER).rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(SOURCE_FOLDER)
            
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # Resize image
                temp_path = Path(TEMP_DIR) / f"img_{count}.jpg"
                resize_image(file_path, temp_path)
                zipf.write(temp_path, rel_path)
                temp_path.unlink()  # Delete temp file
                count += 1
                if count % 200 == 0:
                    print(f"Processed {count}/{total_images} images...")

            elif file_path.suffix.lower() in ['.txt', '.yaml']:
                # Copy labels and yaml directly
                zipf.write(file_path, rel_path)

# Clean up temp folder
shutil.rmtree(TEMP_DIR, ignore_errors=True)

print(f"\nDONE! Created: {OUTPUT_ZIP}")
print("Size: ~12–18 GB → Perfect for Google Colab!")
print("Upload this zip to Google Drive NOW!")
input("Press Enter to exit...")