# augment_pcb_final.py  ← RUN THIS ONE FILE ONLY
import albumentations as A
import cv2
import os
import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET

# --------------------- CHANGE ONLY THESE 2 LINES IF NEEDED ---------------------
RAW_DATA = r"E:\IBA_MS_DS 2026\Computer Vision\Project\PCB_DATASET"
AUG_OUT  = r"E:\IBA_MS_DS 2026\Computer Vision\Project\PCB_DATASET\augmented_full"
# ------------------------------------------------------------------------------

os.makedirs(AUG_OUT, exist_ok=True)
os.makedirs(os.path.join(AUG_OUT, "images"), exist_ok=True)
os.makedirs(os.path.join(AUG_OUT, "Annotations"), exist_ok=True)

defects = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']

# Fixed & safe pipelines (works with Albumentations 1.3+ and 1.4+)
transforms = [
    (A.Compose([
        A.Rotate(limit=18, border_mode=cv2.BORDER_CONSTANT, value=(143,148,151), p=0.9),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomScale(scale_limit=0.25, p=0.7)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3)), "geom"),
    
    (A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.8), "bright"),
    (A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=25, p=0.7), "hsv"),
    (A.GaussNoise(variance_limit=(10, 60), p=0.6), "noise"),  # variance_limit works in all versions
    (A.MotionBlur(blur_limit=7, p=0.4), "blur"),
    (A.CoarseDropout(max_holes=5, max_height=45, max_width=45, fill_value=0, p=0.5), "dropout")
]

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes, labels = [], []
    for obj in root.findall('object'):
        name = obj.find('name').text
        b = obj.find('bndbox')
        boxes.append([int(b.find('xmin').text), int(b.find('ymin').text),
                      int(b.find('xmax').text), int(b.find('ymax').text)])
        labels.append(name)
    return boxes, labels, tree

def save_xml(tree, boxes, labels, img_shape, save_path, filename):
    root = tree.getroot()
    root.find('filename').text = filename
    h, w = img_shape[:2]
    root.find('size/width').text = str(w)
    root.find('size/height').text = str(h)
    for obj in root.findall('object'): root.remove(obj)
    for box, label in zip(boxes, labels):
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = label
        b = ET.SubElement(obj, 'bndbox')
        for name, val in zip(['xmin','ymin','xmax','ymax'], box):
            ET.SubElement(b, name).text = str(int(val))
    tree.write(save_path)

total = 0
for defect in defects:
    img_dir = os.path.join(RAW_DATA, "images", defect)
    xml_dir = os.path.join(RAW_DATA, "Annotations", defect)
    out_img = os.path.join(AUG_OUT, "images", defect)
    out_xml = os.path.join(AUG_OUT, "Annotations", defect)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_xml, exist_ok=True)
    
    images = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.JPG"))
    print(f"\n{defect}: {len(images)} original images → will create ~{len(images)*12} augmented")
    
    for img_path in tqdm(images, desc=defect):
        base = os.path.splitext(os.path.basename(img_path))[0]
        xml_path = os.path.join(xml_dir, base + ".xml")
        if not os.path.exists(xml_path): continue
            
        img = cv2.imread(img_path)
        boxes, labels, tree = parse_xml(xml_path)
        
        # Create 12 versions per image (6 transforms × 2 iterations)
        for iter_id in range(2):
            for i, (tf, name) in enumerate(transforms):
                if name == "geom":
                    aug = tf(image=img, bboxes=boxes, class_labels=labels)
                    aug_img = aug['image']
                    aug_boxes = aug['bboxes'] if aug['bboxes'] else boxes
                    aug_labels = aug['class_labels'] if 'class_labels' in aug else labels
                else:
                    aug = tf(image=img)
                    aug_img = aug['image']
                    aug_boxes, aug_labels = boxes, labels
                
                if aug_img is None: continue
                
                new_name = f"{base}_v{iter_id}_{name}.jpg"
                cv2.imwrite(os.path.join(out_img, new_name), aug_img)
                save_xml(tree, aug_boxes, aug_labels, aug_img.shape,
                        os.path.join(out_xml, new_name.replace(".jpg", ".xml")), new_name)
                total += 1

print(f"\n🎉 SUCCESS! Created {total:,} augmented images + XMLs")
print(f"   → Location: {AUG_OUT}")
print("   → Next step: run prepare_data.py")