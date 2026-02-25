import streamlit as st
import pickle
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from ultralytics import YOLO
import joblib
import time
import os
import shutil
from PIL import Image
import pandas as pd
import io
import base64

# Charger le modèle final
BASE_DIR = Path(__file__).resolve().parent
final_model_path = BASE_DIR / "models" / "final_model.pkl"
with open(final_model_path, 'rb') as file:
    final_model = pickle.load(file)

yolo_model = YOLO(final_model["yolo_model_path"])
svm_model = joblib.load(final_model["svm_model_path"])
resnet_model = final_model["resnet_model"]

# Répertoire pour stocker les images persistantes
PERSISTENCE_DIR = "persistent_images"
if not os.path.exists(PERSISTENCE_DIR):
    os.makedirs(PERSISTENCE_DIR)

# Fonction pour extraire les caractéristiques pour SVM
def extract_features_for_svm(object_image):
    if object_image.shape[2] == 3:
        object_image = cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB)

    object_image = cv2.resize(object_image, (224, 224))
    img_array = image.img_to_array(object_image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = resnet_model.predict(img_array)
    return features.flatten()

# Fonction de classification SVM
def classify_image_svm(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    features = extract_features_for_svm(img_rgb)
    prediction = svm_model.predict([features])
    
    label = "Recyclable" if prediction == 0 else "Organic"
    return img_rgb, label

# Fonction de segmentation et classification avec YOLO et SVM
def process_image_yolo_svm(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    start_time = time.time()
    results = yolo_model.predict(image_path)
    processing_time = time.time() - start_time

    recyclable_count = 0
    organic_count = 0

    boxes = results[0].boxes
    class_ids = boxes.cls
    confidences = boxes.conf

    for i, class_id in enumerate(class_ids):
        x1, y1, x2, y2 = boxes[i].xyxy[0].cpu().numpy()
        object_image = img_rgb[int(y1):int(y2), int(x1):int(x2)]

        features = extract_features_for_svm(object_image)
        prediction = svm_model.predict([features])
        label = "Recyclable" if prediction == 0 else "Organic"

        if label == "Recyclable":
            recyclable_count += 1
        else:
            organic_count += 1
        
        color = (0, 255, 0) if label == "Recyclable" else (255, 0, 0)
        cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img_rgb, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    image_details = {
        "Image": Image.fromarray(img_rgb),  # Convertir l'image en objet PIL
        "Image Path": image_path,
        "Image Size": f"{img.shape[1]}x{img.shape[0]}",
        "Recyclable Detected": recyclable_count,
        "Organic Detected": organic_count,
        "Processing Time": f"{processing_time * 1000:.2f} ms"
    }

    return img_rgb, image_details

# Interface Streamlit
st.title("♻️🌍 Protection de l'Environnement")
st.sidebar.image("https://www.euroschoolindia.com/wp-content/uploads/2023/07/environmental-safety.jpg", width=250)

# Créer la barre latérale pour naviguer entre les pages
page = st.sidebar.radio("Choisir une action", ["Classification avec SVM", "Segmentation avec YOLO et SVM", "Team"])

# Initialiser les variables de session pour stocker les images et les classes
if 'svm_image_data' not in st.session_state:
    st.session_state.svm_image_data = []

if 'yolo_image_data' not in st.session_state:
    st.session_state.yolo_image_data = []

# Page 1 : Classification avec SVM
if page == "Classification avec SVM":
    st.header("Classification avec le modèle SVM")
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "png"])

    if uploaded_file:
        # Sauvegarder l'image téléchargée dans un répertoire persistant
        image_path = os.path.join(PERSISTENCE_DIR, f"{uploaded_file.name}")
        with open(image_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        processed_image, label = classify_image_svm(image_path)
        st.image(processed_image, channels="RGB", caption=f"Classifié comme : {label}")
        
        # Ajouter l'image et la classe au tableau persistant
        st.session_state.svm_image_data.append({"Image": Image.open(uploaded_file), "Class": label})

    # Afficher un tableau des images et de leurs classes pour la classification SVM
    if st.session_state.svm_image_data:
        st.write("Tableau des images et de leurs classes (Classification SVM) :")
        image_data_for_table = []

        # Convertir les images en base64 et préparer les données pour le tableau
        for data in st.session_state.svm_image_data:
            # Convertir l'image en base64
            image_buffer = io.BytesIO()
            data["Image"].save(image_buffer, format="PNG")
            image_base64 = base64.b64encode(image_buffer.getvalue()).decode("utf-8")

            image_data_for_table.append({
                "Image": f'<img src="data:image/png;base64,{image_base64}" width="100"/>',
                "Class": data["Class"]
            })

        # Afficher le tableau avec des images
        df = pd.DataFrame(image_data_for_table)
        st.markdown(df.to_html(escape=False, render_links=True), unsafe_allow_html=True)

# Page 2 : Segmentation avec YOLO et SVM
elif page == "Segmentation avec YOLO et SVM":
    st.header("Segmentation et Classification avec YOLO et SVM")
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "png"])

    if uploaded_file:
        # Sauvegarder l'image téléchargée dans un répertoire persistant
        image_path = os.path.join(PERSISTENCE_DIR, f"{uploaded_file.name}")
        with open(image_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        processed_image, image_details = process_image_yolo_svm(image_path)
        st.image(processed_image, channels="RGB", caption="Image Annotée")


        # Ajouter l'image et sa classe au tableau persistant pour YOLO + SVM
        st.session_state.yolo_image_data.append({
            "Image": Image.open(uploaded_file),
            "Class": f"Recyclable: {image_details['Recyclable Detected']}, Organic: {image_details['Organic Detected']}"
        })

    # Afficher un tableau des images et de leurs classes pour la segmentation YOLO + SVM
    if st.session_state.yolo_image_data:
        st.write("Tableau des images et de leurs classes (Segmentation YOLO + SVM) :")
        image_data_for_table = []

        # Convertir les images en base64 et préparer les données pour le tableau
        for data in st.session_state.yolo_image_data:
            # Convertir l'image en base64
            image_buffer = io.BytesIO()
            data["Image"].save(image_buffer, format="PNG")
            image_base64 = base64.b64encode(image_buffer.getvalue()).decode("utf-8")

            image_data_for_table.append({
                "Image": f'<img src="data:image/png;base64,{image_base64}" width="100"/>',
                "Class": data["Class"]
            })

        # Afficher le tableau avec des images
        df = pd.DataFrame(image_data_for_table)
        st.markdown(df.to_html(escape=False, render_links=True), unsafe_allow_html=True)

# Page 3 : Team
elif page == "Team":
    st.subheader("Team Members Information")

    # URLs des photos des membres
    photo1_url = "https://media.licdn.com/dms/image/v2/D4D03AQHzhvWZztFG9g/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1718304956444?e=1740009600&v=beta&t=1OfdIEvt2-mcKxiRXARxBp8nX8KbWBpdKUT2brcw9_M"
    photo2_url = "https://media.licdn.com/dms/image/v2/D5603AQGUFtCpihVUbw/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1709069836403?e=1740009600&v=beta&t=8LbEepeC5rQh4vlTAveiIdHksGpf7XmYkQ_6wjtlY4s"

    # Affichage des photos côte à côte avec leurs noms en dessous
    col1, _, col2 = st.columns(3)  # Créez trois colonnes pour afficher les photos côte à côte

    with col1:
        st.image(photo1_url, width=200)  # Ajustez la largeur de l'image
        st.markdown("Moheddine BEN ABDALLAH - I3-FSS")

    with col2:
        st.image(photo2_url, width=200)  # Ajustez la largeur de l'image
        st.markdown("Débora GNUITO - I3-FSS")

