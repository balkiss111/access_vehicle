from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from models.recognition import *

def detect_vehicle(img_array, model_vehicle):
    """Détecter le véhicule principal dans l'image"""
    # Initialisation des résultats
    results = {
        "vehicle_detected": False,
        "vehicle_box": None,
        "img_with_boxes": img_array.copy()
    }
    
    # Conversion de format si nécessaire (PIL à numpy)
    if isinstance(img_array, Image.Image):
        img_array = np.array(img_array)
    
    # Détection d'orientation
    height, width = img_array.shape[:2]
    if height > width:  # Portrait, rotation nécessaire
        img_array = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)
        results["img_with_boxes"] = img_array.copy()
    
    # Détection avec le modèle
    results_vehicle = model_vehicle(img_array)
    
    for r in results_vehicle:
        if hasattr(r, 'boxes') and r.boxes:
            # Trouver la plus grande boîte
            largest_box = None
            max_area = 0
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    largest_box = (x1, y1, x2, y2)
            
            if largest_box:
                x1, y1, x2, y2 = largest_box
                results["vehicle_box"] = largest_box
                results["vehicle_detected"] = True
                
                # Dessiner le rectangle
                cv2.rectangle(results["img_with_boxes"], (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(results["img_with_boxes"], "VEHICLE", (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
    
    # Préparer le retour
    message = "Véhicule détecté" if results["vehicle_detected"] else "Aucun véhicule détecté"
    return f"{message} - Vous pouvez maintenant détecter la couleur", Image.fromarray(results["img_with_boxes"])

def detect_color(img_array, model_color, vehicle_box=None, vehicle_detected=False):
    """Détecter la couleur du véhicule"""
    # Conversion de format si nécessaire
    if isinstance(img_array, Image.Image):
        img_array = np.array(img_array)
    
    # Initialisation des résultats
    results = {
        "color_detected": False,
        "label_color": None,
        "img_with_boxes": img_array.copy()
    }
    
    # Détection dans la ROI si véhicule détecté
    if vehicle_detected and vehicle_box:
        x1, y1, x2, y2 = vehicle_box
        vehicle_roi = img_array[y1:y2, x1:x2]
        results_color = model_color(vehicle_roi)
    else:
        results_color = model_color(img_array)
    
    # Traitement des résultats
    for r in results_color:
        if hasattr(r, 'boxes') and r.boxes:
            cls = int(r.boxes.cls[0])
            results["label_color"] = r.names[cls]
            results["color_detected"] = True
            
            # Dessiner les boîtes
            box = r.boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            
            # Ajuster les coordonnées si ROI
            if vehicle_detected and vehicle_box:
                vx1, vy1, vx2, vy2 = vehicle_box
                abs_x1 = vx1 + x1
                abs_y1 = vy1 + y1
                abs_x2 = vx1 + x2
                abs_y2 = vy1 + y2
            else:
                abs_x1, abs_y1, abs_x2, abs_y2 = x1, y1, x2, y2
            
            # Dessiner le rectangle et le texte
            cv2.rectangle(results["img_with_boxes"], (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)
            cv2.putText(results["img_with_boxes"], "COLOR", (abs_x1, abs_y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            if results["label_color"]:
                cv2.putText(results["img_with_boxes"], results["label_color"], 
                           (abs_x1, abs_y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 255), 2)
    
    return f"Couleur: {results['label_color']}" if results["color_detected"] else "Couleur non détectée", Image.fromarray(results["img_with_boxes"])

# [Les autres fonctions restent identiques...]

def detect_orientation(img_array, model_orientation, is_video=False):
    """Détecter l'orientation du véhicule"""
    # Initialisation des résultats
    results = {
        "orientation_detected": False,
        "label_orientation": None,
        "img_with_boxes": img_array.copy(),
        "corrected_orientation": False
    }
    
    # Conversion de format si nécessaire
    if isinstance(img_array, Image.Image):
        img_array = np.array(img_array)
    
    # Correction d'orientation pour les vidéos
    if is_video:
        height, width = img_array.shape[:2]
        if height > width:  # Portrait, rotation nécessaire
            img_array = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)
            results["corrected_orientation"] = True
    
    # Détection avec le modèle
    results_orientation = model_orientation(img_array)
    

    # Traitement des résultats
    for r in results_orientation:
        if hasattr(r, 'boxes') and r.boxes and hasattr(r.boxes, 'cls') and len(r.boxes.cls) > 0:
            cls = int(r.boxes.cls[0])
            results["label_orientation"] = r.names[cls]
            results["orientation_detected"] = True
            
            # Dessiner la boîte de détection
            box = r.boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            
            cv2.rectangle(results["img_with_boxes"], (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(results["img_with_boxes"], "ORIENTATION", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            
            if results["label_orientation"]:
                cv2.putText(results["img_with_boxes"], results["label_orientation"],
                           (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (255, 255, 0), 2)
    
    # Préparer le retour
    if results["orientation_detected"]:
        message = f"Orientation: {results['label_orientation']}"
    else:
        message = "Orientation non détectée"
    
    return message, Image.fromarray(results["img_with_boxes"])

def detect_logo_and_model(img_array, model_logo, model_per_brand, model_labels, cnn_logo_model):
    """Détecter et reconnaître logo et modèle"""
    # Initialisation des résultats
    results = {
        "brand": None,
        "model": "Modèle non détecté",
        "logo_results": [],
        "logo_img": None,
        "img_with_boxes": img_array.copy()
    }
    
    # Conversion de format si nécessaire
    if isinstance(img_array, Image.Image):
        img_array = np.array(img_array)
    
    # Détection des logos
    logo_results = model_logo(img_array)
    
    if logo_results and logo_results[0].boxes:
        for box in logo_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Dessiner la boîte du logo
            cv2.rectangle(results["img_with_boxes"], (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(results["img_with_boxes"], "LOGO", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # Extraire le logo
            logo_crop = img_array[y1:y2, x1:x2]
            results["logo_img"] = Image.fromarray(logo_crop)
            
            # Reconnaissance de la marque
            brand_recognition = recognize_logo(results["logo_img"], cnn_logo_model)
            results["logo_results"].append(brand_recognition)
            
            if not results["brand"] or "confiance" not in str(results["brand"]):
                results["brand"] = brand_recognition
            
            # Reconnaissance du modèle si marque reconnue
            brand_name = None
            if "(" in brand_recognition:
                brand_name = brand_recognition.split("(")[0].strip().lower()
            else:
                brand_name = brand_recognition.lower()
            
            if brand_name in model_per_brand:
                try:
                    results["model"] = recognize_model(
                        brand_name, 
                        results["logo_img"],
                        model_per_brand[brand_name],
                        model_labels[brand_name]
                    )
                    
                    # Ajouter le modèle à l'image
                    cv2.putText(results["img_with_boxes"], f"Modèle: {results['model']}", 
                               (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 255), 2)
                except Exception as e:
                    print(f"Erreur reconnaissance modèle: {str(e)}")
                    results["model"] = "Erreur reconnaissance modèle"
    
    # Détection globale si échec
    if not results["brand"] or any(x in str(results["brand"]) for x in ["incertaine", "Erreur"]):
        global_brand = predict_brand(img_array, cnn_logo_model)
        if global_brand and "non détectée" not in global_brand:
            results["brand"] = global_brand
    
    # Préparer les messages de retour
    brand_msg = f"Marque: {results['brand']}" if results['brand'] else "Marque non détectée"
    model_msg = f"Modèle: {results['model']}" if results['model'] else "Modèle non détecté"
    logo_msg = " | ".join(results["logo_results"]) if results["logo_results"] else "Aucun logo reconnu"
    
    return (
        brand_msg,
        model_msg,
        f"Reconnaissance logo: {logo_msg}",
        Image.fromarray(results["img_with_boxes"]),
        results["logo_img"]
    )

def detect_plate(shared_results, model_plate_detection, model_characters):
    """Détecter la plaque d'immatriculation dans l'image"""
    if shared_results["img_rgb"] is None:
        return "Veuillez d'abord charger une image/vidéo", None, None, None

    shared_results["trocr_char_list"] = []
    shared_results["trocr_combined_text"] = ""
    img_to_process = shared_results["img_rgb"]

    # Utiliser l'image corrigée si nécessaire
    if shared_results.get("corrected_orientation", False):
        height, width = img_to_process.shape[:2]
        if height > width:  # Portrait, besoin de rotation
            img_to_process = cv2.rotate(img_to_process, cv2.ROTATE_90_CLOCKWISE)

    # Si un véhicule a été détecté, utiliser cette zone pour la détection
    if shared_results["vehicle_detected"] and shared_results["vehicle_box"]:
        vx1, vy1, vx2, vy2 = shared_results["vehicle_box"]
        roi = img_to_process[vy1:vy2, vx1:vx2]
        results_plate = model_plate_detection(roi)
    else:
        results_plate = model_plate_detection(img_to_process)

    if results_plate and results_plate[0].boxes:
        for box in results_plate[0].boxes:
            # Ajuster les coordonnées si on a utilisé la ROI du véhicule
            if shared_results["vehicle_detected"] and shared_results["vehicle_box"]:
                vx1, vy1, vx2, vy2 = shared_results["vehicle_box"]
                rx1, ry1, rx2, ry2 = map(int, box.xyxy[0])
                # Convertir en coordonnées absolues
                x1 = vx1 + rx1
                y1 = vy1 + ry1
                x2 = vx1 + rx2
                y2 = vy1 + ry2
            else:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

            shared_results["detection_boxes"]["plate"] = (x1, y1, x2, y2)
            plate_crop = img_to_process[y1:y2, x1:x2]
            shared_results["plate_crop_img"] = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
            break

    return shared_results

def is_empty_plate(cropped_plate_image):
    """Détecte si la plaque est visuellement vide (espace blanc)"""
    if cropped_plate_image is None:
        return True

    # Convertir en numpy array si c'est une image PIL
    if isinstance(cropped_plate_image, Image.Image):
        plate_img = np.array(cropped_plate_image)
    else:
        plate_img = cropped_plate_image

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(plate_img, cv2.COLOR_RGB2GRAY)

    # Seuillage pour détecter les zones non blanches
    _, thresholded = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

    # Compter les pixels non blancs (potentiels caractères)
    non_white_pixels = cv2.countNonZero(thresholded)

    # Si moins de 1% de pixels non blancs, considérer comme vide
    total_pixels = gray.shape[0] * gray.shape[1]
    return non_white_pixels < (0.01 * total_pixels)
