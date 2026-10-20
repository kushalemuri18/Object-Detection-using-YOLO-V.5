import streamlit as st
import torch
import cv2
import tempfile
import os
from PIL import Image
import numpy as np
from pathlib import Path

@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

model = load_model()

def main():
    st.set_page_config(page_title="YOLOv5 Object Detection", layout="wide")
    st.title("🚀 YOLOv5 Object Detection Web App")
    st.write("Upload an image, a video, or use your webcam for real-time object detection.")

 
    st.sidebar.title("Navigation")
    input_source = st.sidebar.radio(
        "Choose Input Source:",
        ("📷 Image", "🎥 Video", "📹 Live Webcam"),
        index=0
    )

    if input_source == "📷 Image":
        st.header("Image Detection")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            with st.spinner('Processing image...'):
                img_array = np.array(image)
                if img_array.shape[-1] == 4:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                
                results = model(img_array)
                results_img = np.squeeze(results.render())
                
                st.subheader("Detection Results")
                st.image(results_img, caption='Processed Image', use_column_width=True)
                
                
                results.save() 
                st.success("✅ Detection output saved to: `runs/detect/exp/`")
                
                st.subheader("Detected Objects")
                detections = results.pandas().xyxy[0]
                if not detections.empty:
                    st.dataframe(detections[['name', 'confidence']])
                else:
                    st.write("No objects detected.")

    elif input_source == "🎥 Video":
        st.header("Video Detection")
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])
        
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name

            st.video(video_path)

            if st.button('▶️ Process Video'):
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error("Error opening video file.")
                    return

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # Create output directory: runs/detect/exp/
                exp_dir = Path("runs/detect/exp")
                exp_dir.mkdir(parents=True, exist_ok=True)
                output_filename = f"processed_{uploaded_file.name}"
                output_path = exp_dir / output_filename

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

                stframe = st.empty()
                progress_bar = st.progress(0)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                current_frame = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame)
                    frame_with_boxes = np.squeeze(results.render())
                    out.write(frame_with_boxes)

                    frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, channels='RGB', use_column_width=True)

                    current_frame += 1
                    progress_bar.progress(current_frame / frame_count)

                cap.release()
                out.release()

                st.success("✅ Video processing complete!")
                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=open(output_path, 'rb').read(),
                    file_name=output_filename,
                    mime='video/mp4'
                )
                st.success(f"✅ Video saved to: `{output_path}`")

    elif input_source == "📹 Live Webcam":
        st.header("Live Webcam Detection")
        st.write("Click below to start your webcam. Object detection will run in real-time.")
        
        run = st.checkbox('▶️ Start Webcam')
        FRAME_WINDOW = st.image([])

        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            st.error("Could not access webcam. Please check permissions.")
            return

        while run:
            ret, frame = camera.read()
            if not ret:
                st.write("Failed to grab frame.")
                break

            results = model(frame)
            frame_with_boxes = np.squeeze(results.render())

            # Save first 10 frames to runs/detect/exp/
            if 'frame_count' not in st.session_state:
                st.session_state.frame_count = 0

            if st.session_state.frame_count < 10:
                exp_dir = Path("runs/detect/exp")
                exp_dir.mkdir(parents=True, exist_ok=True)
                frame_filename = f"webcam_frame_{st.session_state.frame_count:03d}.jpg"
                frame_path = exp_dir / frame_filename
                cv2.imwrite(str(frame_path), frame_with_boxes)
                st.session_state.frame_count += 1

            frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb, channels='RGB')

        else:
            st.write('⏹️ Webcam stopped.')
            camera.release()

if __name__ == '__main__':
    main()