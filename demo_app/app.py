import streamlit as st
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from ultralytics import YOLO
import tempfile
import os
from pathlib import Path
import time

# Page configuration
st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .defect-box {
        border: 2px solid #ff4444;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        background: #fff5f5;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ========================================================
# MODEL LOADING (CACHED)
# ========================================================
@st.cache_resource
def load_model(model_path):
    """Load YOLO model with caching"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ========================================================
# AUGMENTATION PIPELINE
# ========================================================
def get_augmentation_pipeline():
    """Define augmentation transformations"""
    return A.Compose([
        A.Rotate(limit=45, p=0.7, border_mode=cv2.BORDER_REFLECT),
        A.RandomScale(scale_limit=(0.7, 1.4), p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        A.Blur(blur_limit=(3, 7), p=0.4),
        A.HorizontalFlip(p=0.5),
        A.Resize(height=512, width=512, always_apply=True)
    ])

# ========================================================
# IMAGE PROCESSING
# ========================================================
def process_image(image, model, conf_threshold=0.35):
    """Run inference on image"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Ensure RGB format
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Run inference
    results = model(img_array, conf=conf_threshold, imgsz=512, verbose=False)[0]
    
    # Get annotated image
    annotated = results.plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    return results, annotated, img_array

def apply_augmentation(img_array, transform):
    """Apply augmentation to image"""
    augmented = transform(image=img_array)
    return augmented['image']

# ========================================================
# STREAMLIT APP
# ========================================================
def main():
    # Header
    st.markdown('<p class="main-header">🔍 PCB Defect Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Knowledge Distillation YOLOv8n Model | Response-Based KD</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Model path input
    model_path = st.sidebar.text_input(
        "Model Path",
        value="demo_app/Models/best.pt",
        help="Path to your trained YOLOv8 model"
    )
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.sidebar.error(f"❌ Model not found at: {model_path}")
        st.sidebar.info("📁 Please place your model at: `models/best.pt`")
        st.stop()
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model(model_path)
    
    if model is None:
        st.error("Failed to load model. Please check the path.")
        st.stop()
    
    st.sidebar.success("✅ Model loaded successfully!")
    
    # Model info
    with st.sidebar.expander("📊 Model Information"):
        st.write(f"**Architecture:** YOLOv8n (Knowledge Distilled)")
        st.write(f"**Parameters:** ~3M")
        st.write(f"**Classes:** {len(model.names)}")
        st.write(f"**Class Names:** {', '.join(model.names.values())}")
    
    # Detection settings
    st.sidebar.subheader("🎯 Detection Settings")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help="Minimum confidence for detections"
    )
    
    # Augmentation settings
    st.sidebar.subheader("🔄 Augmentation Test")
    enable_augmentation = st.sidebar.checkbox(
        "Enable Robustness Testing",
        value=False,
        help="Test model with augmented versions"
    )
    
    num_augmentations = 3
    if enable_augmentation:
        num_augmentations = st.sidebar.slider(
            "Number of Augmentations",
            min_value=1,
            max_value=5,
            value=3
        )
    
    # Main content
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload PCB Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a PCB image for defect detection"
    )
    
    # Sample images option
    st.markdown("**Or try a sample image:**")
    col1, col2, col3 = st.columns(3)
    
    sample_clicked = None
    if col1.button("Sample PCB 1"):
        sample_clicked = "Test_images/image1.jpg"
    if col2.button("Sample PCB 2"):
        sample_clicked = "Test_images/image2.jpg"
    if col3.button("Sample PCB 3"):
        sample_clicked = "Test_images/image3.jpg"
    
    # Load image
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    elif sample_clicked and os.path.exists(sample_clicked):
        image = Image.open(sample_clicked)
    
    if image is None:
        st.info("👆 Please upload an image or select a sample to begin detection")
        
        # Show example results section
        st.markdown("---")
        st.subheader("📈 Expected Performance")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><h3>92.6%</h3><p>mAP@0.5</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><h3>52.1%</h3><p>mAP@0.5:0.95</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><h3>95.5%</h3><p>Precision</p></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card"><h3>88.3%</h3><p>Recall</p></div>', unsafe_allow_html=True)
        
        st.stop()
    
    # Process image
    st.markdown("---")
    st.subheader("🔬 Detection Results")
    
    # Original image detection
    with st.spinner("Running detection..."):
        start_time = time.time()
        results, annotated_img, img_array = process_image(image, model, conf_threshold)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📷 Original Image**")
        st.image(image, use_column_width=True)
    
    with col2:
        st.markdown("**🎯 Detection Result**")
        st.image(annotated_img, use_column_width=True)
    
    # Detection statistics
    num_detections = len(results.boxes)
    
    st.markdown("---")
    st.subheader("📊 Detection Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Defects Detected", num_detections)
    with col2:
        st.metric("Inference Time", f"{inference_time:.1f} ms")
    with col3:
        st.metric("FPS", f"{1000/inference_time:.1f}")
    with col4:
        status = "❌ Defective" if num_detections > 0 else "✅ Clean"
        st.metric("PCB Status", status)
    
    # Detailed detections
    if num_detections > 0:
        st.markdown("### 🔍 Detected Defects")
        
        for idx, box in enumerate(results.boxes):
            cls_id = int(box.cls.item())
            cls_name = model.names[cls_id]
            confidence = box.conf.item()
            bbox = box.xyxy[0].cpu().numpy()
            
            with st.expander(f"Defect {idx + 1}: {cls_name} ({confidence:.2%})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Class:** {cls_name}")
                    st.write(f"**Confidence:** {confidence:.4f}")
                with col2:
                    st.write(f"**Bounding Box:**")
                    st.write(f"- X1: {bbox[0]:.1f}, Y1: {bbox[1]:.1f}")
                    st.write(f"- X2: {bbox[2]:.1f}, Y2: {bbox[3]:.1f}")
    else:
        st.success("✅ No defects detected! PCB appears to be clean.")
    
    # Augmentation testing
    if enable_augmentation:
        st.markdown("---")
        st.subheader("🔄 Robustness Testing")
        st.info("Testing model performance under various augmentations (rotation, scaling, noise, etc.)")
        
        transform = get_augmentation_pipeline()
        
        aug_cols = st.columns(num_augmentations)
        
        for idx in range(num_augmentations):
            with aug_cols[idx]:
                # Apply augmentation
                aug_img = apply_augmentation(img_array, transform)
                
                # Run detection
                aug_pil = Image.fromarray(aug_img)
                aug_results, aug_annotated, _ = process_image(aug_pil, model, conf_threshold)
                
                st.markdown(f"**Augmented {idx + 1}**")
                st.image(aug_annotated, use_column_width=True)
                
                aug_detections = len(aug_results.boxes)
                st.metric(f"Detections", aug_detections)
                
                if aug_detections > 0:
                    confs = [f"{b.conf.item():.2f}" for b in aug_results.boxes[:3]]
                    st.caption(f"Conf: {', '.join(confs)}")
    
    # Export results
    st.markdown("---")
    st.subheader("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Save annotated image
        if st.button("📥 Download Annotated Image"):
            # Convert to PIL
            annotated_pil = Image.fromarray(annotated_img)
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                annotated_pil.save(tmp_file.name, 'JPEG')
                
                with open(tmp_file.name, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download",
                        data=f.read(),
                        file_name="pcb_detection_result.jpg",
                        mime="image/jpeg"
                    )
    
    with col2:
        # Export detection report
        if st.button("📄 Generate Report"):
            report = f"""
PCB DEFECT DETECTION REPORT
{'='*50}

Image: {uploaded_file.name if uploaded_file else 'Sample Image'}
Detection Time: {inference_time:.2f} ms
Confidence Threshold: {conf_threshold}

RESULTS:
- Total Defects: {num_detections}
- Status: {'DEFECTIVE' if num_detections > 0 else 'CLEAN'}

DETECTED DEFECTS:
"""
            if num_detections > 0:
                for idx, box in enumerate(results.boxes):
                    cls_name = model.names[int(box.cls.item())]
                    conf = box.conf.item()
                    report += f"\n{idx+1}. {cls_name} (Confidence: {conf:.4f})"
            else:
                report += "\nNo defects detected."
            
            report += f"\n\n{'='*50}\nModel: YOLOv8n (Knowledge Distilled)\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            
            st.download_button(
                label="⬇️ Download Report",
                data=report,
                file_name="detection_report.txt",
                mime="text/plain"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>PCB Defect Detection System</strong></p>
        <p>Powered by YOLOv8n with Response-Based Knowledge Distillation</p>
        <p>Model Performance: 92.6% mAP@0.5 | 3M Parameters | ~120 FPS</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
