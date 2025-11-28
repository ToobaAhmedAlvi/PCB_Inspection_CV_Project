# 03_prepare_yolo_dataset.py
import os
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

def voc_to_yolo(xml_path, out_txt_path, class_map_lower):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        w = int(root.find('size/width').text)
        h = int(root.find('size/height').text)

        with open(out_txt_path, 'w') as f:
            for obj in root.findall('object'):
                label = obj.find('name').text.strip()        # Get label
                label_lower = label.lower()                   # Normalize to lowercase
                
                if label_lower not in class_map_lower:
                    print(f"Warning: Unknown label '{label}' in {xml_path} → skipping")
                    continue
                    
                class_id = class_map_lower[label_lower]
                
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Convert to YOLO format
                x_center = (xmin + xmax) / (2 * w)
                y_center = (ymin + ymax) / (2 * h)
                width = (xmax - xmin) / w
                height = (ymax - ymin) / h
                
                # Clip to prevent errors
                x_center = max(min(x_center, 1.0), 0.0)
                y_center = max(min(y_center, 1.0), 0.0)
                width = max(min(width, 1.0), 0.0)
                height = max(min(height, 1.0), 0.0)
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")

if __name__ == "__main__":
    # PATHS
    data_path = r"E:\\IBA_MS_DS 2026\\Computer Vision\\Project\\PCB_DATASET\\augmented_full\\images"
    ann_path  = r"E:\\IBA_MS_DS 2026\\Computer Vision\\Project\\PCB_DATASET\\augmented_full\\Annotations"
    out_path  = r"E:\\IBA_MS_DS 2026\\Computer Vision\\Project\\PCB_DATASET\\yolo_dataset"

    # CLASS MAP — lowercase keys = bulletproof!
    class_map = {
        'missing_hole': 0,
        'mouse_bite': 1,
        'open_circuit': 2,
        'short': 3,
        'spur': 4,
        'spurious_copper': 5
    }

    os.makedirs(out_path, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(out_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(out_path, split, 'labels'), exist_ok=True)

    all_images = []
    for defect_folder in os.listdir(data_path):
        img_dir = os.path.join(data_path, defect_folder)
        ann_dir = os.path.join(ann_path, defect_folder)
        
        if not os.path.isdir(img_dir):
            continue
            
        for img_file in os.listdir(img_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(img_dir, img_file)
                xml_path = os.path.join(ann_dir, os.path.splitext(img_file)[0] + ".xml")
                
                if os.path.exists(xml_path):
                    all_images.append((img_path, xml_path, defect_folder))
                else:
                    print(f"Missing XML: {xml_path}")

    print(f"Found {len(all_images)} valid image-XML pairs")

    # Stratified split
    defects = [defect_folder.lower() for _, _, defect_folder in all_images]
    train_val, test = train_test_split(all_images, test_size=0.10, stratify=defects, random_state=42)
    train, val = train_test_split(train_val, test_size=0.15, stratify=[d.lower() for _, _, d in train_val], random_state=42)

    splits = [(train, 'train'), (val, 'val'), (test, 'test')]
    
    for split_data, split_name in splits:
        print(f"Processing {split_name} split → {len(split_data)} images")
        for img_path, xml_path, _ in split_data:
            # Copy image
            dst_img = os.path.join(out_path, split_name, 'images', os.path.basename(img_path))
            shutil.copy(img_path, dst_img)
            
            # Convert annotation
            txt_path = os.path.join(out_path, split_name, 'labels', os.path.basename(img_path).rsplit('.', 1)[0] + '.txt')
            voc_to_yolo(xml_path, txt_path, class_map)
    
    print("YOLO dataset created successfully!")
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")