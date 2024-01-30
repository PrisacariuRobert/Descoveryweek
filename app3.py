# Import the libraries
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
from keras.models import load_model

# Load the model and the labels

model = load_model("app/keras_model.h5", compile=False)
class_names = open("app/labels.txt", "r").readlines()
favicon_path = "app/favicon.ico"
st.set_page_config(page_title="Image Classifier", page_icon=favicon_path)

# Define a function to preprocess the image
def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

# Define a function to get the prediction and the confidence score
def get_prediction(image):
    data = preprocess_image(image)
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score

# Create a title for the app
st.title("Image Classifier ")

# Create a container to display the webcam
st.markdown("## Webcam")
webcam_container = st.empty()

# Display initial text for prediction
prediction_text = st.text("")

# Initialize the container for additional information
information_container = st.empty()

# Dictionary to store class pop-up messages
class_messages = {
    0: "To conserve leopards, it is important to work with communities that live near them. The African Wildlife Foundation works closely with pastoralist communities to institute preventative measures to protect livestock from predation. In Tanzania, AWF builds bomas for communities living in close proximity to carnivores. These are predator-proof enclosures keep livestock safe from carnivores. By taking proactive steps we are able to prevent both livestock and carnivore deaths. Another solution is to use Global Positioning System (GPS) collars to study leopards. AWF believes the key to ensuring the future of the leopard lies in an integrated approach to conservation that looks not only at the species itself but at the needs of local people, land use, and the ecosystem as a whole.",
    1: "To conserve lions, it is important to work with communities that live near them. The African Wildlife Foundation works closely with pastoralist communities to institute preventative measures to protect livestock from predation. In Tanzania, AWF builds bomas for communities living in close proximity to carnivores. These are predator-proof enclosures keep livestock safe from carnivores. By taking proactive steps we are able to prevent both livestock and carnivore deaths. Another solution is to use Global Positioning System (GPS) collars to study lions. AWF believes the key to ensuring the future of the lion lies in an integrated approach to conservation that looks not only at the species itself but at the needs of local people, land use, and the ecosystem as a whole.",
    2: "To conserve elephants, it is important to work with communities that live near them. The African Wildlife Foundation works closely with pastoralist communities to institute preventative measures to protect livestock from predation. In Tanzania, AWF builds bomas for communities living in close proximity to carnivores. These are predator-proof enclosures keep livestock safe from carnivores. By taking proactive steps we are able to prevent both livestock and carnivore deaths. Another solution is to use Global Positioning System (GPS) collars to study elephants. AWF believes the key to ensuring the future of the elephant lies in an integrated approach to conservation that looks not only at the species itself but at the needs of local people, land use, and the ecosystem as a whole.",
    3: "To conserve rhinos, it is important to work with communities that live near them. Save the Rhino International works to conserve all five rhino species, by supporting rhino conservation programmes across Africa and Asia. They also work with local communities to help them develop sustainable livelihoods that do not depend on poaching or other illegal activities 2. Another solution is to use Global Positioning System (GPS) collars to study rhinos. Save the Rhino International believes the key to ensuring the future of the rhino lies in an integrated approach to conservation that looks not only at the species itself but at the needs of local people, land use, and the ecosystem as a whole.",
    4: "To conserve African buffalos, it is important to work with communities that live near them. The African Wildlife Foundation works with government entities to help plan and propose alternative solutions to habitat fragmentation by providing its scientists as resources to assist in proper planning to ensure a balance between growth and modernization and wildlife conservation. They also work with communities to help meet their agricultural needs through proper planning and techniques for sustainable agricultural growth. By providing these resources, AWF is able to minimize land"
}

# Open the webcam using OpenCV
cap = cv2.VideoCapture(0)

# Move fps definition outside the loop
fps = st.sidebar.slider("Frames per second", 1, 30, 10)
with st.expander("Camera Settings"):
    # Move width and height definition inside the expander
    width_key = "width_slider" + str(np.random.randint(0, 100000))
    height_key = "height_slider" + str(np.random.randint(0, 100000))

    width = st.slider("Width", 100, 800, 400, key=width_key)
    height = st.slider("Height", 100, 800, 400, key=height_key)

# Loop until the user stops the app

# Loop until the user stops the app
while True:


    ret, frame = cap.read()
    if not ret:
        st.error("Error reading frame from webcam.")
        break
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (width, height))
    image = Image.fromarray(frame)
    webcam_container.image(image, channels="RGB")
    class_name, confidence_score = get_prediction(image)
    prediction_text.text(f"Prediction: {class_name[2:]} | Confidence Score: {confidence_score:.2f}")

    class_index = np.argmax(model.predict(preprocess_image(image)))
    if class_index in class_messages:
        # Display the additional information
        information_container.markdown(class_messages[class_index])

    cv2.waitKey(int(1000/fps))

cap.release()
