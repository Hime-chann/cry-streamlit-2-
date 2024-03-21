import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import librosa
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import pickle

# Define the Streamlit app
def app():
    st.title('Infant Cry Classification')

    # Display choice of classifier
    options = ['LSTM', 'Random Forest']
    selected_option = st.selectbox('Select the classifier', options)

    # Define model loading functions based on classifier type
    def load_lstm_model():
        try:
            model_path = "lstm_audio_model.joblib"
            model = joblib.load(model_path)
            return model
        except FileNotFoundError:
            st.error(f"LSTM model not found at '{model_path}'. Please ensure the model exists.")
            return None

    def load_random_forest_model():
        try:
            model_path = "myRandomForest.pkl"
            with open(model_path, "rb") as file:
                model = pickle.load(file)
            return model
        except FileNotFoundError:
            st.error(f"Random Forest model not found at '{model_path}'. Please ensure the model exists.")
            return None

    if selected_option == 'Random Forest':
        model = load_random_forest_model()
    else:
        model = load_lstm_model()
        if model is None:
            st.warning("Model loading failed. Classification functionality unavailable.")

    # Real-time audio recording function
    def record_audio():
        fs = 44100  # Sample rate
        seconds = 5  # Duration of recording

        st.write("Recording...")
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished

        st.write("Recording finished.")
        return myrecording, fs

    # Function to predict cry from audio file
    def predict_cry(audio_file):
        try:
            # Preprocess audio (extract MFCC features)
            audio, sr = librosa.load(audio_file)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=1)
            mfcc_scaled = StandardScaler().fit_transform(mfcc.T)
            mfcc_df = pd.DataFrame(mfcc_scaled.T)

            prediction = model.predict(mfcc_df)

            # Get the class label
            class_names = model.classes_
            predicted_class = class_names[prediction[0]]
            return predicted_class
        except Exception as e:
            st.error("Error occurred during prediction.")
            return None

    # User interface
    uploaded_audio = st.file_uploader("Upload audio file (WAV format)", type=["wav"])

    if uploaded_audio is not None:
        with open("uploaded_audio.wav", "wb") as f:
            f.write(uploaded_audio.getbuffer())

    if st.button("Record Audio"):
        recording, fs = record_audio()
        wav.write('recorded_audio.wav', fs, recording)
        st.audio('recorded_audio.wav', format='audio/wav')

    if uploaded_audio is not None or st.button("Classify"):
        if uploaded_audio is None:
            st.error("Please upload an audio file.")
        else:
            predicted_cry = predict_cry("uploaded_audio.wav")
            if predicted_cry is not None:
                st.success(f"Predicted cry: {predicted_cry}")

# Run the app
if __name__ == "__main__":
    app()
            
