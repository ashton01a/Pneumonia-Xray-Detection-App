# 🩻 Pneumonia Detection from Chest X-Rays | AI Deep Learning App

This project builds and deploys a deep learning web app that classifies chest X-ray images as **Normal** or **Pneumonia**.  
Built with **TensorFlow**, **Streamlit**, and deployed live on **Streamlit Cloud**.

---

## 🌟 Live Demo

👉 [Try the App Here!](https://pneumonia-xray-detection-app-jmsr8fgdsvbpmfbxoqq64n.streamlit.app/)  
_(Replace this with your real app link once ready!)_

---

## 📂 Project Structure

| File | Description |
|:-----|:------------|
| `app.py` | Streamlit web app source code |
| `requirements.txt` | Python packages for deployment |
| `pneumonia_detector.h5` | Deep learning model (commented out for cloud testing) |

---

## 🛠 Built With

- **Python** 🐍
- **TensorFlow** 🔥
- **OpenCV** 📷
- **Streamlit** 🚀

---

## 📈 Skills Demonstrated

- Deep Learning Model Deployment
- Image Preprocessing and Augmentation
- Frontend/Backend AI App Integration
- GitHub, Version Control, and CI/CD Practices
- Debugging Cloud Deployment Issues

---

## 🛠 Model Compression and Accuracy Note

This application uses a **compressed TensorFlow Lite (TFLite) version** of the pneumonia detection model for faster cloud deployment.  
Due to compression optimizations, **prediction confidence may vary slightly** from original model performance.

**Goal:** Prioritize speed, lightweight deployment, and user accessibility in cloud environments like Streamlit Cloud.

---

## 🚀 Future Improvements

- 🔥 Retrain a lightweight pneumonia detection model using **Quantization-Aware Training (QAT)** for improved TFLite accuracy.
- 🔥 Integrate **Grad-CAM** heatmaps to visualize which regions of the X-ray influenced the prediction.
- 🔥 Add batch prediction support for uploading multiple X-ray files at once.
- 🔥 Explore deploying a containerized version using **Docker** and **Render.com** to support larger models and scalable hosting.
- 🔥 Implement detailed prediction reports and download options for clinicians or users.

---

## 👨‍💻 Author

- **Ashton Jones**  
- [LinkedIn Profile](https://linkedin.com/in/ashton01a)

---
