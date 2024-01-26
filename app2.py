import streamlit as st
from tensorflow.keras.models import load_model

from PIL import Image, ImageOps
import numpy as np
import cv2
import time

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

class_names = open("labels.txt", "r").readlines()

def classify_image(img):
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(img)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    return class_name, confidence_score

def webcam_capture():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Unable to open the webcam.")
        return

    recording = st.checkbox("Start Recording")

    while recording:
        ret, frame = cap.read()

        # Check if the frame is empty
        if not ret:
            st.error("Error: Unable to capture frame from the webcam.")
            break

        # Convert frame to RGB for PIL processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the webcam feed
        st.image(rgb_frame, caption="Webcam Feed", use_column_width=True)

        pil_image = Image.fromarray(rgb_frame)
        class_name, confidence_score = classify_image(pil_image)

        # Create a rounded border with a black background around the class name and confidence percentage
        st.markdown(
            f"""
            <div style="background-color: black; border: 2px solid black; border-radius: 10px; padding: 10px; margin-top: 10px;">
                <p style="font-size: 18px; color: white;"><b>Class:</b> {class_name}</p>
                <p style="font-size: 18px; color: white;"><b>Confidence Score:</b> {confidence_score * 100:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Announce the recognized class
        st.success(f"Recognized class: {class_name} with probability: {confidence_score * 100:.2f}%")

        # Check if the confidence score is above a certain threshold (adjust as needed)
        threshold = 0.9  # Adjust as needed
        if confidence_score > threshold:
            break  # Break out of the loop if confidence score is high enough

        # Sleep for a short duration to avoid high CPU usage
        time.sleep(0.1)

    # Release the webcam when the loop exits
    cap.release()

def main():
    st.title("Webcam Image Classification App")
    st.markdown("""
        #### Real-time Image Classification using Webcam
        This app captures images from your webcam and classifies them using a pre-trained model.
    """)
    webcam_capture()

if __name__ == "__main__":
    main()
