import streamlit as st
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

st.set_page_config(page_title="Parking Spot Detection", page_icon="🅿️", layout="centered")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="RawanAlwadeya/YOLOParkingDetection",
        filename="YOLOParkingDetection.pt"
    )
    return YOLO(model_path)

model = load_model()


def detect_parking(model, image: Image.Image, conf=0.25, imgsz=640):
    """
    Detect parking spots in an uploaded image using YOLO.
    Returns counts of empty and occupied spots, and the prediction image.
    """

    img_array = np.array(image.convert("RGB"))

    results = model.predict(img_array, conf=conf, imgsz=imgsz, verbose=False)

    img_bgr = results[0].plot()
    img_rgb = img_bgr[:, :, ::-1]
    pred_img = Image.fromarray(img_rgb)

    
    pred_classes = results[0].boxes.cls.cpu().numpy().astype(int)
    names = model.names
    labels = [names[c] for c in pred_classes]

    empty_count = labels.count("empty")
    occupied_count = labels.count("occupied")

    return empty_count, occupied_count, pred_img



# st.set_page_config(page_title="Parking Spot Detection", page_icon="🅿️", layout="centered")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Detection"])


if page == "Home":
    st.markdown("<h1 style='text-align: center;'>🅿️ Parking Spot Detection App</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Fine-tuned YOLOv8 Model</h3>", unsafe_allow_html=True)

    st.write(
        """
        Efficient use of **parking spaces** is a major challenge in urban environments.  
        This app leverages a **YOLOv8 model fine-tuned on parking data** to detect whether parking spots are:
        
        - ✅ **Empty**  
        - 🚗 **Occupied**

        The model highlights each parking spot in the uploaded image, allowing for **automated monitoring** of parking availability.  
        """
    )

    st.image(
        "https://img.freepik.com/premium-photo/cars-parked-row-outdoor-parking-back-view-3d-illustration_926199-2459067.jpg",
        caption="Parking Area Example",
        use_container_width=True
    )

    st.info("👉 Go to the **Detection** page from the left sidebar to upload a parking lot image and get predictions.")


elif page == "Detection":
    st.markdown("<h1 style='text-align: center;'>🅿️ Parking Spot Detection</h1>", unsafe_allow_html=True)
    st.write(
        "Upload an image of a parking area, and the model will detect whether spots are **empty** or **occupied**."
    )

    uploaded_file = st.file_uploader("Upload Parking Lot Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        empty_count, occupied_count, pred_img = detect_parking(model, image)

        st.image(pred_img, caption="Model Prediction", use_container_width=True)

        st.info(
            f"🅿️ **Empty spots:** {empty_count}\n\n"
            f"🚗 **Occupied spots:** {occupied_count}"
        )
