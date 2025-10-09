# **YOLOParkingDetection: Fine-Tuned YOLOv8 for Parking Spot Detection**

YOLOParkingDetection is a deep learning project that uses a **YOLOv8 architecture** to detect parking spot availability in images, classifying each spot as **Empty** or **Occupied**.  
It demonstrates a complete **end-to-end computer vision workflow** including **dataset preparation, EDA, YOLOv8 fine-tuning, evaluation, and deployment with Streamlit & Hugging Face**.

---

## **Demo**

- 🎥 [View LinkedIn Demo Post](https://www.linkedin.com/posts/rawan-alwadeya-17948a305_deeplearning-yolov8-computervision-activity-7379997449509199873-c2fq?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAAE3YzG0BAZw48kimDDr_guvq8zXgSjDgk_I)  
- 🌐 [Try the App Live on Streamlit](https://yoloparkingdetection-yeqnqxdi7ev7zrm9ketsyk.streamlit.app/)  
- 🤗 [Explore on Hugging Face](https://huggingface.co/RawanAlwadeya/YOLOParkingDetection)  

![App Demo](https://github.com/rawan-alwadiya/YOLOParkingDetection/blob/main/YOLOParkingDetection%20App.png)  
![Parking Detection Example](https://github.com/rawan-alwadiya/YOLOParkingDetection/blob/main/Parking%20Detection.png)

---

## **Project Overview**

The workflow includes:  
- **Exploration & Visualization**: Inspected dataset distribution and bounding boxes  
- **Modeling (YOLOv8)**: Transfer learning from YOLOv8 pretrained on the COCO dataset  
- **Evaluation**: Precision, Recall, mAP metrics for robust performance tracking  
- **Deployment**: Real-time interactive **Streamlit web app** and **Hugging Face Spaces**  

---

## **Objective**

Develop and deploy a robust **object detection model** to support **smart parking management systems**, enabling real-time monitoring of available and occupied parking spaces in urban environments.

---

## **Dataset**

- **Content**: Parking lot images with bounding boxes for each parking spot  
- **Labels**:  
  - **Empty**  
  - **Occupied**  
- **Preparation**: Data cleaning, annotation, and splitting into train/validation/test sets  

---

## **Project Workflow**

- **EDA & Visualization**: Explored dataset structure and bounding box distribution  
- **Transfer Learning**: Fine-tuned YOLOv8 on the parking dataset    
- **Deployment**: Streamlit application and Hugging Face model hosting  

---

## **Performance Results**

**YOLOv8 Parking Detection Model:**  
- **Precision**: `99.39%`  
- **Recall**: `98.20%`  
- **mAP@50**: `98.73%`  
- **mAP@50-95**: `97.46%`  
- **Speed**: ~10.7ms per image (real-time capable)  

These results confirm the model’s effectiveness for real-world parking availability monitoring.

---

## **Tech Stack**

**Languages & Libraries**:  
- Python, Pandas, NumPy  
- PyTorch, Ultralytics YOLOv8  
- OpenCV, Matplotlib, Seaborn  
- Streamlit (Deployment)

---

## **👩‍💻 Author**

**Rawan Alwadeya**  
AI Engineer | Generative AI Engineer | Data Scientist  
- 📧 Email: r.wadia21@gmail.com 
- 🌐 [LinkedIn Profile](https://www.linkedin.com/in/rawan-alwadeya-17948a305/) 

---

⚡ With this project, AI-driven **smart parking detection** can support urban mobility, optimize space usage, and reduce traffic congestion.
