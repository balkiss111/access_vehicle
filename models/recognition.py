from PIL import Image
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os 

# Classes prédéfinies pour la reconnaissance de logo
LOGO_CLASSES = sorted([
    d for d in os.listdir('models/train') 
    if os.path.isdir(os.path.join('models/train', d))
])

def preprocess_image(image, target_size):
    """Prétraiter une image pour la classification"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 2:  # Si image en niveaux de gris
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # Si image RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:  # Si image BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    resized = cv2.resize(image, target_size)
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)

def recognize_logo(cropped_logo, cnn_logo_model, logo_classes=LOGO_CLASSES):
    """Reconnaître la marque à partir d'un logo détecté"""
    try:
        if cropped_logo.size == 0:
            return "Logo trop petit pour analyse"

        # Pré-traitement
        input_img = preprocess_image(cropped_logo, (128, 128))
        
        # Prédiction
        predictions = cnn_logo_model.predict(input_img, verbose=0)
        pred_index = np.argmax(predictions[0])
        pred_label = logo_classes[pred_index]
        pred_conf = predictions[0][pred_index]

        return f"{pred_label} ({pred_conf:.2f})" if pred_conf >= 0.5 else f"Marque incertaine: {pred_label} ({pred_conf:.2f})"
        
    except Exception as e:
        print(f"Erreur reconnaissance logo: {str(e)}")
        return "Erreur d'analyse"

def recognize_model(brand, logo_crop, model_recognizer, model_labels):
    """Reconnaître le modèle spécifique d'une voiture"""
    try:
        if logo_crop.size == 0:
            return "Image trop petite pour analyse"

        # Pré-traitement spécifique au modèle
        input_shape = model_recognizer.input_shape[1:3]
        input_img = preprocess_image(logo_crop, (input_shape[1], input_shape[0]))

        # Prédiction
        predictions = model_recognizer.predict(input_img, verbose=0)
        return model_labels[np.argmax(predictions[0])]
        
    except Exception as e:
        print(f"Erreur reconnaissance modèle: {str(e)}")
        return "Erreur de détection"

def predict_brand(image, cnn_logo_model, logo_classes=LOGO_CLASSES):
    """Prédire la marque globale du véhicule à partir de l'image complète"""
    try:
        input_img = preprocess_image(image, (224, 224))
        predictions = cnn_logo_model.predict(input_img, verbose=0)
        pred_index = np.argmax(predictions[0])
        confidence = predictions[0][pred_index]
        
        if confidence < 0.5:
            return "Marque non détectée (confiance trop faible)"
        return f"{logo_classes[pred_index]} (confiance: {confidence:.2f})"
        
    except Exception as e:
        print(f"[ERREUR] Prédiction de marque: {str(e)}")
        return "Erreur de détection"


def recognize_characters(shared_results, model_characters, trocr_processor, trocr_model):
    """Reconnaître les caractères sur la plaque détectée"""
    if "plate_crop_img" not in shared_results or shared_results["plate_crop_img"] is None:
        return shared_results

    plate_crop = np.array(shared_results["plate_crop_img"])
    plate_for_char_draw = plate_crop.copy()

    # Détection des caractères
    results_chars = model_characters(plate_crop)
    char_boxes = []
    for r in results_chars:
        if r.boxes:
            for box in r.boxes:
                x1c, y1c, x2c, y2c = map(int, box.xyxy[0])
                char_boxes.append(((x1c, y1c, x2c, y2c), x1c))

    char_boxes.sort(key=lambda x: x[1])

    for i, (coords, _) in enumerate(char_boxes):
        x1c, y1c, x2c, y2c = coords
        char_crop = plate_crop[y1c:y2c, x1c:x2c]
        char_pil = Image.fromarray(char_crop).convert("RGB")

        try:
            inputs = trocr_processor(images=char_pil, return_tensors="pt").pixel_values
            generated_ids = trocr_model.generate(inputs)
            predicted_char = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            shared_results["trocr_char_list"].append(predicted_char)
        except Exception as e:
            shared_results["trocr_char_list"].append("?")

        cv2.rectangle(plate_for_char_draw, (x1c, y1c), (x2c, y2c), (255, 0, 255), 1)
        cv2.putText(plate_for_char_draw, predicted_char, (x1c, y1c - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)

    shared_results["plate_with_chars_img"] = Image.fromarray(cv2.cvtColor(plate_for_char_draw, cv2.COLOR_BGR2RGB))
    shared_results["trocr_combined_text"] = ''.join(shared_results["trocr_char_list"])
    
    return shared_results
