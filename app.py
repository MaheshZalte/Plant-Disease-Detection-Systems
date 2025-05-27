import streamlit as st
# import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import os
import logging
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Plant Disease Recognition",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Session state defaults
if "app_mode" not in st.session_state:
    st.session_state["app_mode"] = "home"
if "feedback_data" not in st.session_state:
    st.session_state["feedback_data"] = []
if "contact_data" not in st.session_state:
    st.session_state["contact_data"] = []

# Load model once
@st.cache_resource
def load_model():
    try:
        custom_objects = {'InputLayer': tf.keras.layers.InputLayer}
        return tf.keras.models.load_model("PlantDisease_Model.h5", custom_objects=custom_objects)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

def load_css():
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sigmar&family=Roboto:wght@400;700&display=swap');
    h1, h2, h3, h4, h5, h6 { font-family: 'Sigmar', cursive; color: #2E86C1; }
    body, p, div, span, input, button, textarea, label, select { font-family: 'Roboto', sans-serif; }
    .stButton>button { background-color: #27AE60; color: #FFFFFF; border: none; border-radius: 8px; padding: 0.6em 1.2em; font-size: 1rem; cursor: pointer; transition: background-color 0.3s ease, color 0.3s ease; }
    .stButton>button:hover { background-color: #2ECC71; color: #000000; }
    div[data-testid="stProgressBar"] > div > div { background-color: #2ECC71 !important; }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
load_css()

def get_base64_image(image_path: str) -> str:
    if not os.path.isfile(image_path):
        logging.warning(f"File not found or is a directory: {image_path}")
        return ""
    try:
        with open(image_path, "rb") as image_file:
            encoded_bytes = base64.b64encode(image_file.read())
            return encoded_bytes.decode("utf-8")
    except Exception as e:
        logging.error(f"Error reading file {image_path}: {e}")
        return ""

background_image = get_base64_image("bag.jpg")
if background_image:
    page_bg_img = f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.3),rgba(0, 0, 0, 0.3)), url("data:image/jpg;base64,{background_image}") no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Disease info dictionary (same as before)
disease_info = {
    "Pepper_bell_Bacterial_spot": {
        "plant": "Pepper Bell",
        "solution": "Use copper-based bactericides and avoid overhead watering."
    },
    "Pepper_bell_healthy": {
        "plant": "Pepper Bell",
        "solution": "The plant is healthy. Maintain proper watering and nutrient levels."
    },
    "Potato_Early_blight": {
        "plant": "Potato",
        "solution": "Use fungicides with chlorothalonil and practice crop rotation."
    },
    "Potato_healthy": {
        "plant": "Potato",
        "solution": "The plant is healthy. Ensure good soil drainage."
    },
    "Potato_Late_blight": {
        "plant": "Potato",
        "solution": "Use copper-based fungicides and remove affected leaves."
    },
    "Tomato_Target_Spot": {
        "plant": "Tomato",
        "solution": "Apply fungicides like chlorothalonil and avoid excess moisture."
    },
    "Tomato_Tomato_mosaic_virus": {
        "plant": "Tomato",
        "solution": "Remove infected plants immediately and control aphids."
    },
    "Tomato_Tomato_YellowLeaf_Curl_Virus": {
        "plant": "Tomato",
        "solution": "Control whiteflies and use virus-resistant varieties."
    },
    "Tomato_Bacterial_spot": {
        "plant": "Tomato",
        "solution": "Use copper sprays and avoid overhead irrigation."
    },
    "Tomato_Early_blight": {
        "plant": "Tomato",
        "solution": "Use chlorothalonil-based fungicides regularly."
    },
    "Tomato_healthy": {
        "plant": "Tomato",
        "solution": "The plant is healthy. Maintain good cultivation practices."
    },
    "Tomato_Late_blight": {
        "plant": "Tomato",
        "solution": "Apply fungicides containing mancozeb and remove infected leaves."
    },
    "Tomato_Leaf_Mold": {
        "plant": "Tomato",
        "solution": "Ensure proper ventilation and use fungicides if needed."
    },
    "Tomato_Septoria_leaf_spot": {
        "plant": "Tomato",
        "solution": "Remove infected leaves and use copper-based fungicides."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "plant": "Tomato",
        "solution": "Use neem oil or insecticidal soaps to control mites."
    }
}

def process_image(image_data):
    if not image_data:
        st.error("No image uploaded.")
        return None
    try:
        image = Image.open(image_data).convert("RGB")
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        arr = np.array(image) / 255.0
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        st.error(f"🔥 Error Processing Image: {e}")
        return None

def model_prediction(image_data):
    if model is None:
        st.error("⚠️ Model not loaded. Check the model file.")
        return None, None
    processed = process_image(image_data)
    if processed is None:
        return None, None
    try:
        preds = model.predict(processed)
        idx = int(np.argmax(preds))
        conf = float(np.max(preds))
        if idx >= len(disease_info):
            st.error("⚠️ Prediction out of bounds.")
            return None, None
        return idx, conf
    except Exception as e:
        st.error(f"🔥 Prediction Error: {e}")
        return None, None

def home_page():
    st.markdown("""
        <div class="welcome-section">
            <h1>🌿 Welcome to Plant Disease Recognition</h1>
            <p>Your AI-powered assistant for detecting and treating plant diseases</p>
        </div>
    """, unsafe_allow_html=True)
    st.image("home_page.jpeg", use_container_width=True)
    st.markdown("""
        <div class="features-grid">
            <div class="feature-card">
                <h3>🔍 Disease Detection</h3>
                <p>Upload leaf images for instant disease analysis using advanced AI technology.</p>
            </div>
            <div class="feature-card">
                <h3>💡 Smart Solutions</h3>
                <p>Get personalized treatment recommendations and prevention tips.</p>
            </div>
            <div class="feature-card">
                <h3>📊 Detailed Reports</h3>
                <p>Access comprehensive analysis with confidence scores and insights.</p>
            </div>
            <div class="feature-card">
                <h3>🌱 Plant Health</h3>
                <p>Monitor and maintain the health of your plants with expert guidance.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div class="stats-section">
            <div class="stat-card">
                <h4>15+</h4>
                <p>Plant Diseases Detected</p>
            </div>
            <div class="stat-card">
                <h4>98%</h4>
                <p>Detection Accuracy</p>
            </div>
            <div class="stat-card">
                <h4>24/7</h4>
                <p>Available Support</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    # Sidebar Navigation
    st.sidebar.title("Plant Disease Detection System")
    nav_options = {
        "Disease Recognition": ("🔍", "disease"),
        "About": ("ℹ️", "about"),
        "Feedback": ("📝", "feedback"),
        "View Feedback": ("📋", "view_feedback")
    }
    for label, (icon, page) in nav_options.items():
        if st.sidebar.button(f"{icon} {label}", key=f"nav_{page}"):
            st.session_state["app_mode"] = page
            st.rerun()

def disease_recognition_page():
    st.markdown('<h1 class="page-title">🔍 Disease Recognition</h1>', unsafe_allow_html=True)
    test_image = st.file_uploader("📸 Upload a leaf image", type=["jpg", "png", "jpeg"])
    if test_image:
        st.image(test_image, caption="Preview", use_container_width=False)
    else:
        st.markdown("""
            <div class="uploadfile">
                <h3>Drag and drop your image here</h3>
                <p>Supported formats: JPG, PNG, JPEG</p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("""
        <div class="instructions">
            <h3>📝 How to Use</h3>
            <ul>
                <li>Upload a clear, well-lit image of the plant leaf</li>
                <li>Ensure the affected area is visible in the image</li>
                <li>Click "Analyze" to detect any diseases</li>
                <li>View detailed results and treatment recommendations</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    if test_image:
        if st.button("🔍 Analyze Image"):
            with st.spinner("🔄 Analyzing your image..."):
                result_index, confidence = model_prediction(test_image)
            if result_index is not None:
                class_names = list(disease_info.keys())
                disease_name = class_names[result_index]
                plant_name = disease_info[disease_name]["plant"]
                solution = disease_info[disease_name]["solution"]
                st.markdown(f"<div class='result-card'><h3>🌿 Plant Identified</h3><p style='color: #E0E0E0;'>{plant_name}</p></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='result-card'><h3>🔍 Diagnosis</h3><p style='color: #E0E0E0;'>{disease_name.replace('_', ' ')}</p></div>", unsafe_allow_html=True)
                st.markdown("<h3 style='color: #FFFFFF; margin-top: 1rem;'>Confidence Score</h3>", unsafe_allow_html=True)
                st.progress(confidence)
                st.markdown(f"<p style='color: #E0E0E0; text-align: center;'>{confidence:.1%}</p>", unsafe_allow_html=True)
                with st.expander("💡 Treatment Recommendations"):
                    st.markdown(f"<div style='color: #E0E0E0;'><p>{solution}</p></div>", unsafe_allow_html=True)
                st.balloons()
            else:
                st.error("⚠️ Could not analyze the image. Please try again with a clearer photo.")
    if st.button("← Back to Home"):
        st.session_state["app_mode"] = "home"
        st.rerun()

def about_page():
    st.markdown('<h1 class="about-title">ℹ️ About This Project</h1>', unsafe_allow_html=True)
    image_path = "about_banner.jpeg"
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning("Banner image not found. Please ensure 'about_banner.jpeg' is in the correct location.")
    st.markdown("""
    <div class="about-content">
        This project leverages <b>deep learning</b> to identify plant diseases from leaf images. By analyzing thousands of plant samples, our AI model delivers <b>accurate disease predictions</b> along with confidence scores and recommended treatments.
        <h3>🔍 Key Features:</h3>
        <ul>
            <li>✅ Trained on an extensive dataset of plant leaf images</li>
            <li>✅ Provides precise disease predictions with high accuracy</li>
            <li>✅ Suggests effective treatment methods for various plant diseases</li>
            <li>✅ Simple and user-friendly interface for easy access</li>
        </ul>
        With the power of AI, we aim to help farmers and agriculturists <b>detect diseases early</b> and take <b>preventive actions</b> to ensure healthy crops.
    </div>
    """, unsafe_allow_html=True)
    with st.expander("References & Further Reading"):
        st.markdown(
            """
            - [FAO: Plant Health](http://www.fao.org/plant-health/en/)
            - [Research Paper: Deep Learning for Plant Disease Detection](https://arxiv.org/abs/1604.03169)
            - [GitHub Repository](https://github.com/your-username/plant-disease-recognition) *(if applicable)*
            """
        )
    st.markdown("""
    <div class="info-box">
        We welcome contributions and feedback! Feel free to reach out or submit pull requests.
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go Back", key="go_back"):
        st.session_state["app_mode"] = "home"
        st.rerun()

def feedback_page():
    st.markdown('<h1 class="feedback-title">Feedback & Suggestions</h1>', unsafe_allow_html=True)
    st.markdown('<p class="feedback-content">We value your feedback! Please share your thoughts on how we can improve.</p>', unsafe_allow_html=True)
    user_email = "Guest"
    st.info(f"You are using the app as: **{user_email}**")
    rating = st.slider("On a scale of 1-5, how would you rate your overall experience?", min_value=1, max_value=5, value=3)
    feedback_text = st.text_area("Please share any additional comments or suggestions:")
    name = st.text_input("Your Name (optional)")
    if st.button("Submit Feedback", key="submit_feedback"):
        if feedback_text.strip():
            st.session_state["feedback_data"].append({
                "email": user_email,
                "name": name,
                "rating": rating,
                "comments": feedback_text,
                "timestamp": datetime.utcnow()
            })
            st.success("Thank you for your feedback!")
            st.balloons()
        else:
            st.error("Please enter some feedback before submitting.")
    contact_name = st.text_input("Your Name", key="contact_name")
    contact_email = st.text_input("Your Email", value="", key="contact_email")
    contact_message = st.text_area("Your Message", key="contact_message")
    if st.button("Send Message", key="send_message"):
        if not contact_name.strip() or not contact_email.strip() or not contact_message.strip():
            st.error("Please fill out all fields before sending.")
        else:
            st.session_state["contact_data"].append({
                "name": contact_name,
                "email": contact_email,
                "message": contact_message,
                "timestamp": datetime.utcnow()
            })
            st.success("Thank you! Your message has been sent.")
            st.balloons()
    if st.button("Go Back", key="go_back"):
        st.session_state["app_mode"] = "home"
        st.rerun()

def view_feedback_page():
    st.title("📋 View Feedback & Messages")
    st.markdown("## Feedback")
    feedback_count = 0
    for feedback in st.session_state["feedback_data"]:
        st.markdown(f"""
        <div class="feedback-card">
            <h4>{feedback.get('name', 'Anonymous')}</h4>
            <p><strong>Email:</strong> {feedback.get('email')}</p>
            <p><strong>Rating:</strong> {feedback.get('rating')}</p>
            <p><strong>Comments:</strong> {feedback.get('comments')}</p>
            <p class="timestamp"><strong>Timestamp:</strong> {feedback.get('timestamp').strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)
        feedback_count += 1
    if feedback_count == 0:
        st.info("No feedback available.")
    st.markdown("## Messages")
    contact_count = 0
    for contact in st.session_state["contact_data"]:
        st.markdown(f"""
        <div class="contact-card">
            <h4>{contact.get('name')}</h4>
            <p><strong>Email:</strong> {contact.get('email')}</p>
            <p><strong>Message:</strong> {contact.get('message')}</p>
            <p class="timestamp"><strong>Timestamp:</strong> {contact.get('timestamp').strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)
        contact_count += 1
    if contact_count == 0:
        st.info("No messages available.")
    if st.button("Go Back"):
        st.session_state["app_mode"] = "home"
        st.rerun()

def main():
    mode = st.session_state["app_mode"]
    if mode == "home":
        home_page()
    elif mode == "disease":
        disease_recognition_page()
    elif mode == "about":
        about_page()
    elif mode == "feedback":
        feedback_page()
    elif mode == "view_feedback":
        view_feedback_page()
    else:
        st.session_state["app_mode"] = "home"
        st.rerun()

if __name__ == "__main__":
    main()