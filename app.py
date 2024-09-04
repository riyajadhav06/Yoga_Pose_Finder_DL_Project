import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image

# Load the pre-trained model
model = load_model('yoga.h5')

# Define class names
class_names = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']

# Function to preprocess the image
def load_and_preprocess_image(img, target_size=(75, 75)):
    img = img.resize(target_size)  # Resize image
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image array
    return img_array

# Streamlit app
st.title("Yoga Pose Detection")

# File uploader to allow users to upload images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image for model prediction
    img_array = load_and_preprocess_image(img)
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    
    # Display the prediction
    st.write(f"The predicted yoga pose is: **{predicted_class}**")


from PIL import Image
import numpy as np

# Assuming img_array is already created and resized, insert this code:
if img_array.shape[-1] == 4:  # Check if the image has 4 channels (RGBA)
    img = Image.fromarray((img_array[0] * 255).astype(np.uint8))
    img = img.convert('RGB')  # Convert RGBA to RGB
    img_array = np.array(img).reshape((1, img_array.shape[1], img_array.shape[2], 3))

# Now proceed to make the prediction
prediction = model.predict(img_array)

# import streamlit as st
# import numpy as np


# from tensorflow.keras.models import load_model
# from PIL import Image

# # Load the trained model
# model = load_model('yoga.h5')

# # Define the class names
# class_names = ['downdog', 'goddess', 'plank', 'tree', 'warrior']

# # Function to load and preprocess the image
# def load_and_preprocess_image(image_file):
#     img = Image.open(image_file)
#     img = img.resize((75, 75))
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# # Streamlit app
# st.title('Yoga Pose Detection')

# # File uploader for image input
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
#     # Preprocess the image and make a prediction
#     img_array = load_and_preprocess_image(uploaded_file)
#     predictions = model.predict(img_array)
#     predicted_class = class_names[np.argmax(predictions)]
    
#     # Display the predicted class
#     st.write(f"Predicted Yoga Pose: {predicted_class}")
