# ♻️ Image-Based Garbage Classification Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-Web%20UI-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-T4%20GPU-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

An end-to-end Computer Vision project that automatically classifies waste images into four recyclable categories—**Glass, Plastic, Paper, and Metal**—using Transfer Learning with **MobileNetV2** and an interactive **Gradio** web interface.

---

## 👨‍💻 Authors & Academic Details

* **Institution:** Jalpaiguri Government Engineering College (JGEC), West Bengal, India
* **Department & Semester:** Department of Information Technology — 5th Semester Curriculum Project
* **Submitted by:**
  * **SK AMIR HOSSAIN** — Roll No: `24101106031`
  * **PRATIK PATRA** — Roll No: `24101106032`
* **Submitted to & Supervised by:** Prof. Somnath Dhibar *(Project Mentor & Evaluator)*

---

## 📌 Project Overview

Effective waste sorting is a critical step in modern recycling and environmental sustainability. Manual sorting is slow, labor-intensive, and prone to human error. This project demonstrates how Deep Learning and Convolutional Neural Networks (CNNs) can automate waste identification with high accuracy.

The entire pipeline—from automated dataset downloading and preprocessing to model training, fine-tuning, and web deployment—runs seamlessly in the cloud using **Google Colab** without requiring local GPU resources or manual file setups.

---

## ✨ Key Features

* **Fully Automated Pipeline:** Automatically downloads, extracts, filters, and splits the dataset directly within the notebook—no API keys or local storage required.
* **Transfer Learning & Fine-Tuning:** Uses Google's pre-trained **MobileNetV2** architecture, followed by targeted fine-tuning of the top 20 layers to achieve high accuracy (**85%–87%** on unseen test data).
* **Data Augmentation:** Applies real-time random rotations, shifts, and horizontal flips during training to prevent overfitting and improve generalization.
* **Interactive Web Interface:** Deploys a clean **Gradio** web app where users can upload garbage images and immediately view confidence scores across all four classes.
* **Cloud-Native Execution:** Runs 100% on Google Colab's remote GPU, CPU, and RAM infrastructure, leaving the user's local machine untouched.

---

## 🛠️ Technology Stack

* **Language:** Python 3
* **Deep Learning Framework:** TensorFlow & Keras
* **Model Architecture:** MobileNetV2 (ImageNet weights)
* **Web UI:** Gradio
* **Data Handling & Visualization:** `split-folders`, Pillow (PIL), Matplotlib, NumPy
* **Environment:** Google Colab (NVIDIA T4 GPU)

---

## 📊 Dataset

This project uses the publicly available **TrashNet** dataset.

* **Classes Used (4):** `Glass`, `Plastic`, `Paper`, `Metal` *(filtered from the original 6 classes for balanced, high-accuracy classification).*
* **Image Dimensions:** Resized to $224 \times 224$ RGB pixels to match MobileNetV2 input requirements.
* **Data Split:**
  * **Training Set:** 70% (with Data Augmentation)
  * **Validation Set:** 15% (used for tuning & early stopping)
  * **Test Set:** 15% (held out strictly for final evaluation)

---

## 🧠 Model Architecture & Training Strategy

1. **Base Model:** Pre-trained `MobileNetV2` (`include_top=False`, `weights='imagenet'`).
2. **Custom Classification Head:**
   * `GlobalAveragePooling2D()`
   * `Dropout(0.2)` to reduce overfitting
   * `Dense(4, activation='softmax')` for 4-class probability output
3. **Phase 1 – Feature Extraction:** Base model weights are frozen; only the custom head is trained for up to 10 epochs (`Adam`, learning rate = $0.001$) with `EarlyStopping` and `ModelCheckpoint`.
4. **Phase 2 – Fine-Tuning:** The top 20 layers of MobileNetV2 are unfrozen and trained for 5 additional epochs at a very low learning rate ($10^{-5}$) to refine high-level feature extraction.

---

## 📈 Results & Model Performance

| Metric | Score |
| :--- | :--- |
| **Final Test Accuracy** | `85%–87%` |
| **Final Test Loss** | `0.4135` |

---
## 📁 Project Structure

```text
garbage-classification/
├── garbage_classification.ipynb               # Main Google Colab notebook (training + Gradio UI)
├── README.md                                  # Project documentation
├── requirements.txt                           # Python dependencies
├── test_images/                               # Sample test images for quick evaluation
│   ├── glass_sample.jpg
│   ├── metal_sample.jpg
│   ├── paper_sample.jpg
│   └── plastic_sample.jpg
├── project_report/                            # Academic project report (PDF + Overleaf LaTeX source)
│   ├── garbage_classification_report__.pdf    # Final compiled project report (PDF)
│   ├── main.tex                               # Main LaTeX entry point
│   ├── preamble.tex                           # Packages, layout, and styling
│   ├── titlepage.tex                          # Title page layout
│   ├── references.tex                         # Bibliography / references
│   ├── gradio_screenshot.png                  # Gradio web UI prediction screenshot
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── problem.tex
│   ├── literature.tex
│   ├── requirements.tex
│   ├── technologies.tex
│   ├── dataset.tex
│   ├── methodology.tex
│   ├── implementation.tex
│   ├── results.tex
│   ├── comparison.tex
│   ├── limitations.tex
│   ├── future.tex
│   └── conclusion.tex
└── (Generated in Colab Runtime)
    ├── dataset-resized/                       # Filtered TrashNet dataset (glass, plastic, paper, metal)
    ├── dataset-split/
    │   ├── train/                             # 70% training images
    │   ├── val/                               # 15% validation images
    │   └── test/                              # 15% test images
    └── garbage_classifier.keras               # Saved best fine-tuned MobileNetV2 model
```



### Notebook Sections

| Section | Description |
| :--- | :--- |
| **1. Setup & Dataset Download** | Installs dependencies (`split-folders`, `gradio`), downloads TrashNet, filters to 4 classes, and displays sample images. |
| **2. Data Preprocessing & Splitting** | Splits data (70/15/15) and sets up Keras `ImageDataGenerator` pipelines with MobileNetV2 preprocessing. |
| **3. Model Definition & Training** | Builds the MobileNetV2 transfer learning model, runs initial training, and performs fine-tuning while saving the best weights (`garbage_classifier.keras`). |
| **4. Model Evaluation** | Loads the best saved checkpoint and evaluates final accuracy on the unseen test dataset. |
| **5. Web User Interface (Gradio)** | Launches an interactive image-upload UI embedded in Colab for real-time garbage classification. |

---

## 🚀 How to Run (Google Colab)

You do not need to install Python or TensorFlow on your computer to run this project.

1. Open the project notebook (`garbage_classification.ipynb`) in **[Google Colab](https://colab.research.google.com/)**.
2. Enable GPU acceleration:
   * Navigate to **Runtime** > **Change runtime type**.
   * Select **T4 GPU** under *Hardware accelerator* and click **Save**.
3. Run the entire notebook via **Runtime** > **Run all**.
4. Wait for the dataset to download and the model to train (typically 3–5 minutes on a T4 GPU).
5. Scroll to the final cell (**Section 5: Web User Interface**) to use the embedded Gradio app:
   * Upload an image of glass, plastic, paper, or metal from your device (or use the samples in `test_images/`).
   * Click **Submit** to view the predicted category and confidence percentages.


---

## 📚 Acknowledgments & Citation

The dataset used in this project is sourced from the open-source **TrashNet** repository. If you reference or build upon this work, please credit the original creators:

> Thung, G., & Yang, M. (2016). *Classification of trash for recyclability status*. CS229 Project Report, Stanford University.
