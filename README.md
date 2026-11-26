# ♻️ Image-Based Garbage Classification

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-green.svg)
![Gradio](https://img.shields.io/badge/UI-Gradio-yellow.svg)
![Google Colab](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00.svg)

---

## 🎓 Authors & Academic Details

* **Institution:** Jalpaiguri Government Engineering College (JGEC), West Bengal, India
* **Department & Semester:** Department of Information Technology — 5th Semester Curriculum Project
* **Submitted by:**
  * **SK AMIR HOSSAIN** — Roll No: `24101106031`
  * **PRATIK PATRA** — Roll No: `24101106032`
* **Submitted to & Supervised by:**Prof. Somnath Dhibar (Project Mentor & Evaluator)

---

## 📌 Project Overview

This project implements an automated **Image-Based Garbage Classification** system using Deep Learning and Transfer Learning. Built with **MobileNetV2** in TensorFlow/Keras and deployed with an interactive **Gradio** web interface inside Google Colab, the model classifies uploaded waste images into four recyclable categories:

1. **Glass**
2. **Plastic**
3. **Paper**
4. **Metal**

The goal is to demonstrate a lightweight, accurate, and easy-to-use computer vision pipeline suitable for smart waste-sorting applications.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Base Architecture:** MobileNetV2 (Pre-trained on ImageNet)
* **Image Processing:** Pillow (PIL), NumPy
* **User Interface:** Gradio (Embedded in Google Colab)
* **Development Environment:** Google Colab

---

## 📂 Project Structure

```text
garbage-classification/
├── Garbage_Classification.ipynb   # Main Google Colab notebook (training + Gradio UI)
├── README.md                      # Project documentation
│
└── (Generated in Google Colab during runtime)
    ├── dataset/                   # Extracted TrashNet image dataset
    │   ├── glass/
    │   ├── metal/
    │   ├── paper/
    │   └── plastic/
    └── garbage_classifier.h5      # Saved trained MobileNetV2 model
```

---

## 🧠 Model Architecture & Workflow

1. **Data Preprocessing:** Images are resized to `224 × 224`, converted to RGB, and normalized using `tf.keras.applications.mobilenet_v2.preprocess_input`.
2. **Transfer Learning:** MobileNetV2 (without the top classification layer) is used as the feature extractor, followed by Global Average Pooling, Dropout, and a 4-class Softmax output layer.
3. **Inference Interface:** A clean Gradio interface accepts an uploaded image and outputs confidence scores across all four waste classes.

---

## 🚀 How to Run in Google Colab

1. Open `Garbage_Classification.ipynb` in **Google Colab**.
2. Go to **Runtime > Change runtime type** and select **T4 GPU** for faster training.
3. Run all cells sequentially from top to bottom (**Runtime > Run all**).
4. Once the final cell runs (`interface.launch(share=False, debug=True)`), scroll down to use the embedded Gradio app directly inside the notebook.
5. Upload any image of glass, plastic, paper, or metal to view the predicted class and confidence scores.


---

## 📊 Dataset & Attribution

This project uses the **TrashNet** dataset. Full credit goes to the original creators:

* **Authors:** Gary Thung and Mindy Yang
* **Dataset:** [TrashNet (GitHub)](https://github.com/garythung/trashnet)
