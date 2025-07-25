import streamlit as st
import os
import librosa
import numpy as np
import tensorflow as tf
from itertools import groupby
from tensorflow.keras.models import load_model

# --- App Configuration ---
st.set_page_config(page_title="Capuchinbird Call Detector", layout="wide")

# --- Model & Function Loading ---

# Use Streamlit's caching to load the model only once
@st.cache_resource
def load_keras_model():
    """Load the pre-trained Keras model."""
    try:
        model = load_model('best_capuchin_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Helper functions for audio processing
def load_mp3_16k_mono_librosa(filename):
    """Loads and preprocesses an MP3 file using librosa."""
    wav, _ = librosa.load(filename, sr=16000, mono=True)
    return tf.convert_to_tensor(wav, dtype=tf.float32)

def preprocess_mp3(sample, index=None):
    """Converts a single audio chunk into a spectrogram."""
    sample = sample[0]
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([sample, zero_padding], axis=0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

def classify_audio_file(filepath, model, threshold=0.99):
    """
    Takes a file path and a trained model, and returns the number of 
    Capuchinbird call detections.
    """
    wav = load_mp3_16k_mono_librosa(filepath)
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(
        wav, wav,
        sequence_length=48000,
        sequence_stride=48000,
        batch_size=1
    )
    audio_slices = audio_slices.map(preprocess_mp3)
    audio_slices = audio_slices.batch(64)
    predictions = model.predict(audio_slices)
    binary_preds = [1 if pred > threshold else 0 for pred in predictions]
    detection_count = tf.math.reduce_sum([k for k, _ in groupby(binary_preds)]).numpy()
    
    return {
        "file": os.path.basename(filepath),
        "predictions_per_3s_chunk": binary_preds,
        "total_capuchin_calls_detected": detection_count
    }

# --- Main Application UI ---
st.title("🐦 Capuchinbird Call Detector")
st.write("Upload a forest audio recording (MP3 or WAV) to detect the number of distinct Capuchinbird calls.")

# Load the model
model = load_keras_model()

if model is not None:
    uploaded_file = st.file_uploader("Choose an audio file...", type=['mp3', 'wav'])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with st.spinner('Analyzing audio... This might take a moment.'):
            temp_dir = "temp"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            temp_filepath = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Perform classification
            results = classify_audio_file(temp_filepath, model)
            
            # Display results
            st.success("Analysis complete!")
            st.metric(
                label="Total Capuchinbird Calls Detected", 
                value=int(results['total_capuchin_calls_detected'])
            )

            with st.expander("Show detailed analysis per 3-second chunk"):
                st.write("The model analyzes the audio in 3-second segments.")
                st.write(f"**Predictions (1 = Call Detected, 0 = No Call):**")
                st.text(results['predictions_per_3s_chunk'])

            # Clean up the temporary file
            os.remove(temp_filepath)
else:
    st.warning("Model could not be loaded. Please ensure 'best_capuchin_model.keras' is in the project folder.")