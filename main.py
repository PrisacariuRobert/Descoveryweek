import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("C:\DiscoveryWeek\keras_model.h5", compile=False)

# Load the labels
class_names = open("C:\DiscoveryWeek\labels.txt", "r").readlines()

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
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name[2:], confidence_score

def main():
    st.title("Image Classification App")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        class_name, confidence_score = classify_image(image)

        # Print prediction and confidence score
        st.write("Class:", class_name)
        st.write("Confidence Score:", confidence_score)

if __name__ == "__main__":
    main()

