# 🐦 Capuchinbird Audio Classification Project

A comprehensive, end-to-end machine learning project to detect the calls of the Capuchinbird from raw forest audio recordings. This repository demonstrates a full project lifecycle, from data preprocessing and model training to sophisticated evaluation and deployment as an interactive web application.

![Streamlit Demo](https://github.com/user-attachments/assets/053850cf-a628-4713-a6e5-0ab2b0b5d5e0)


---

## ✨ Core Features

-   **Efficient Data Pipeline**: Utilizes `tf.data.Dataset` for high-performance loading and preprocessing of audio files.
-   **Audio-to-Image Conversion**: Transforms raw audio waves into spectrograms, allowing the use of powerful Convolutional Neural Network (CNN) architectures.
-   **Robust CNN Model**: A custom CNN built with TensorFlow/Keras, featuring Batch Normalization and Dropout layers to ensure stable training and prevent overfitting.
-   **In-Depth Model Evaluation**: Goes beyond simple accuracy to evaluate the model with a **Confusion Matrix** and a **Classification Report** (Precision, Recall, F1-Score), providing a clear picture of its real-world performance.
-   **Interactive Web Application**: A user-friendly UI built with **Streamlit** that allows for easy drag-and-drop prediction on new audio files.
-   **Reproducible & Deployable**: The entire application is containerized with **Docker**, allowing anyone to run it with a single command, ensuring perfect reproducibility.

---

## 💻 Tech Stack

-   **Backend & Modeling**: Python, TensorFlow, Keras, Librosa, Scikit-learn
-   **Frontend & UI**: Streamlit
-   **Deployment**: Docker
-   **Data Manipulation**: NumPy, Matplotlib, Seaborn

---

## 🚀 How to Run This Project

### 1. Run the Web App with Docker (Recommended)

**Prerequisite:** Before running the app, you must first have the trained model file. Run the `AudioClassification.ipynb` notebook to train the model and generate the `best_capuchin_model.keras` file in the root directory.

Once the model file exists, you can build and run the app.
```bash
# 1. Clone this repository
git clone [YOUR GITHUB REPOSITORY URL]
cd [YOUR REPOSITORY NAME]

# 2. Build the Docker image
# This command builds the environment with all dependencies.
docker build -t audio-classifier-app .

# 3. Run the application
# This command starts the Streamlit server inside the container.
docker run -p 8501:8501 audio-classifier-app
