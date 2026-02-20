import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io

# --- CONFIGURATION ---
MODEL_FILE = "vibration_model.h5"
CLASS_NAMES_FILE = "class_names.txt"
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Define the classes and descriptions for rich output
CLASS_DESCRIPTIONS = {
    "class1": {"name": "Constricted Ellipse", "fault": "High Unbalance / Asymmetric Stiffness", "description": "Orbit is very thin and congested, indicating high stiffness or unbalance."},
    "class2": {"name": "Normal Ellipse", "fault": "Standard Asymmetric Stiffness", "description": "Standard elliptical shape indicating normal machine condition or common asymmetric stiffness."},
    "class3": {"name": "Circular/Large Diameter", "fault": "High Unbalance / Resonant Speed", "description": "Large, nearly circular shape, often due to high unbalance or proximity to resonance."},
    "class4": {"name": "Sharp Spikes (Unfiltered)", "fault": "Dent on Probe Tracking Area", "description": "Sharp, sudden edges or spikes coming out of the main orbit, usually caused by a physical dent or scratch on the shaft surface."},
    "class5A": {"name": "Oil Whirl (Multiple Dots)", "fault": "Oil Whirl / Whip Chaos", "description": "Elliptical shape with more than one Key Phasor dot (black dots) indicating fluid-induced instability and chaos."},
    "class5B": {"name": "Fractional Components (Loop + Dots)", "fault": "Fractional Components / Subsynchronous", "description": "The orbit has small internal loops and multiple Key Phasor dots, common in subsynchronous vibration."},
    "class6": {"name": "Banana/Truncation", "fault": "Shaft Rubs / Misalignment", "description": "The orbit exhibits a flatness or truncation on one side (a 'banana' shape), a key sign of intermittent shaft rubbing or severe misalignment."},
    "class7": {"name": "Inner Figure-8", "fault": "Oil-Film Whirl", "description": "The orbit has an internal figure-8 or loop pattern, typically related to oil instability in the bearing."},
    "class8": {"name": "Outer Figure-8", "fault": "Shaft Misalignment", "description": "Classic figure-8 shape, usually indicating severe shaft misalignment."},
    "class9A": {"name": "Petal Shape", "fault": "Subsynchronous Whirl", "description": "Flower-like shape with three or more distinct lobes or 'petals', often seen in complex fractional whirl."},
    "class9B": {"name": "5 Petal Orbit", "fault": "Reverse Precession Harmonic", "description": "Five outward petals. Reverse precession dominated by high harmonic."},
    "class9C": {"name": "Inward Double Loop", "fault": "Forward Precession Harmonic", "description": "Two inward loops. Forward precession harmonic vibration."}
}

@st.cache_resource
def load_model_and_names():
    """Loads the Keras model and class names, caching them for fast re-runs."""
    try:
        model = tf.keras.models.load_model(MODEL_FILE)
    except Exception as e:
        st.error(f"Error loading model '{MODEL_FILE}'. Please ensure 'train.py' was run successfully.")
        st.code(f"Details: {e}")
        return None, None

    try:
        with open(CLASS_NAMES_FILE, 'r') as f:
            class_names = f.read().strip().split(',')
    except Exception as e:
        st.error(f"Error loading class names from '{CLASS_NAMES_FILE}'.")
        st.code(f"Details: {e}")
        return model, None 
        
    return model, class_names

# --- PREDICTION FUNCTION ---

def predict_image(model, class_names, uploaded_file):
    """Processes the image and returns the prediction result."""
    
    image = Image.open(uploaded_file).convert('RGB')

    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image) / 255.0  

    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_index]
    
    confidence = float(predictions[0][predicted_index])

    return predicted_class_name, confidence

# --- STREAMLIT UI ---

def main():
    st.set_page_config(
        page_title="Orbit Classifier",
        layout="centered",
        initial_sidebar_state="auto"
    )
    
    st.title("⚙️ Orbit Fault Classifier")
    st.markdown("Upload a orbit plot (generated or real) to classify the machine's condition based on 10 common fault types.")
    
    # Load Model and Class Names
    model, class_names = load_model_and_names()

    if model is None or class_names is None:
        st.warning("Cannot proceed without the necessary model files.")
        st.stop()
    
    # --- File Uploader ---
    uploaded_file = st.file_uploader(
        "Choose an Orbit Plot Image...", 
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        # --- Display Image and Prediction Button ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption='Uploaded Orbit Plot', use_column_width=True)

        with col2:
            if st.button('Analyze Orbit Plot', type="primary", use_container_width=True):
                # Run prediction when button is clicked
                with st.spinner('Analyzing image and predicting fault type...'):
                    # The confidence returned here is now a Python float
                    predicted_class, confidence = predict_image(model, class_names, uploaded_file)

                # --- Display Results ---
                
                # Retrieve the full description
                result_data = CLASS_DESCRIPTIONS.get(predicted_class, {
                    "name": "Unknown", 
                    "fault": "N/A", 
                    "description": "Prediction result is outside known classes."
                })

                st.subheader(f"✅ Classified Fault")
            
                st.markdown(f"**Class:** `{predicted_class}` - **{result_data['name']}**")
                st.metric(label="Primary Fault Type", value=result_data['fault'])
                st.progress(confidence, text=f"Confidence: {confidence*100:.2f}%")
                st.info(result_data['description'])
            
                st.caption(f"Predicted class index: {class_names.index(predicted_class)}")

    # --- Reference Sidebar ---
    with st.sidebar:
        st.header("Fault Class Reference")
        st.markdown("The model classifies the plot into one of these categories:")
        
        for key, data in CLASS_DESCRIPTIONS.items():
            st.markdown(f"**{key.upper()}** - *{data['fault']}*")
            st.caption(f"{data['description']}")
            st.markdown("---")
            
        st.info("The training script ('train.py') combines data from clean and realistic simulations for robustness.")


if __name__ == '__main__':
    main()