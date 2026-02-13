import streamlit as st
from ultralytics import YOLO
import cv2
import os
import tempfile
import subprocess
from PIL import Image

st.set_page_config(page_title="Intelligent Vision System", page_icon="🚗", layout="wide")

st.title("🚗 Intelligent Autonomous Vehicle Vision System")
st.markdown("### Powered by Optimized YOLOv8 (mAP: 68.8%)")
st.markdown("Upload an image or a video below to detect vehicles and pedestrians in real-time. Use the sidebar to adjust settings.")
st.markdown("---")

@st.cache_resource
def load_model():
    model_path = 'runs/detect/runs/train/YOLOv8s_Advanced_Run/weights/best.pt'
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

model = load_model()

st.sidebar.header("⚙️ Model Configuration")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold (حساسیت مدل)", 
    min_value=0.0, max_value=1.0, value=0.25, step=0.05
)
st.sidebar.info(
    "💡 **Tip:** If the model misses objects, lower the threshold. If it detects fake objects, increase it."
)
st.sidebar.markdown("---")
st.sidebar.markdown("Developed for Phase 2 Demo")

if model is None:
    st.error("❌ Error: Model weights ('best.pt') not found! Please check the path.")
else:
    tab1, tab2 = st.tabs(["🖼️ Image Detection", "🎥 Video Detection"])

    with tab1:
        st.subheader("Image Analysis")
        uploaded_img = st.file_uploader("Choose an image (Traffic, Dashcam, etc.)...", type=['jpg', 'jpeg', 'png', 'webp'])
        
        if uploaded_img is not None:
            image = Image.open(uploaded_img)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Image**")
                st.image(image, use_container_width=True)
            
            if st.button("🔍 Run Object Detection", type="primary"):
                with st.spinner("Analyzing image..."):

                    results = model.predict(source=image, conf=conf_threshold)
                    res_plotted = results[0].plot()
                    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    final_image = Image.fromarray(res_rgb)
                    
                with col2:
                    st.markdown("**Detection Result (Hover to Zoom ⤢)**")
                    st.image(final_image, use_container_width=True)
                
                st.success("✅ Detection Complete! Click the arrows icon on the top right of the image to view it in full screen.")

    with tab2:
        st.subheader("Video Analysis")
        uploaded_vid = st.file_uploader("Choose a video (mp4, avi)...", type=['mp4', 'avi', 'mov'])
        
        if uploaded_vid is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_vid.read())
            tfile.close()
            
            st.video(tfile.name)
            
            if st.button("🎥 Analyze Video", type="primary"):
                with st.spinner("Processing video frame by frame... Please wait."):
                    save_dir = "runs/ui/video_output"
                    os.makedirs(save_dir, exist_ok=True)
                    
                    model.predict(
                        source=tfile.name, 
                        conf=conf_threshold, 
                        save=True, 
                        project=save_dir, 
                        name="predict", 
                        exist_ok=True
                    )
                    
                    output_vid_path = os.path.join(save_dir, "predict", os.path.basename(tfile.name))
                    web_friendly_path = os.path.join(save_dir, "predict", "web_ready.mp4")
                    
                    command = f"ffmpeg -y -i {output_vid_path} -vcodec libx264 {web_friendly_path}"
                    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    if os.path.exists(web_friendly_path):
                        st.markdown("**AI Processed Video (Hover to Fullscreen ⤢)**")
                        st.video(web_friendly_path)
                        st.success("✅ Video processing complete! Use the fullscreen icon on the player.")
                    else:
                        st.warning("Video processed, but web rendering failed.")