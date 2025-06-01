import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import re


# Dictionnaires de référence
CATEGORIES = {
    '1': "Véhicules de tourisme",
    '2': "Camions",
    '3': "Camionnettes",
    '4': "Autocars et autobus",
    '5': "Tracteurs routiers",
    '6': "Autres tracteurs",
    '7': "Véhicules spéciaux",
    '8': "Remorques et semi-remorques",
    '9': "Motocyclettes"
}

WILAYAS = {
    "01": "Adrar", "02": "Chlef", "03": "Laghouat", "04": "Oum El Bouaghi",
    "05": "Batna", "06": "Béjaïa", "07": "Biskra", "08": "Béchar",
    "09": "Blida", "10": "Bouira", "11": "Tamanrasset", "12": "Tébessa",
    "13": "Tlemcen", "14": "Tiaret", "15": "Tizi Ouzou", "16": "Alger",
    "17": "Djelfa", "18": "Jijel", "19": "Sétif", "20": "Saïda",
    "21": "Skikda", "22": "Sidi Bel Abbès", "23": "Annaba", "24": "Guelma",
    "25": "Constantine", "26": "Médéa", "27": "Mostaganem", "28": "MSila",
    "29": "Mascara", "30": "Ouargla", "31": "Oran", "32": "El Bayadh",
    "33": "Illizi", "34": "Bordj Bou Arreridj", "35": "Boumerdès",
    "36": "El Tarf", "37": "Tindouf", "38": "Tissemsilt", "39": "El Oued",
    "40": "Khenchela", "41": "Souk Ahras", "42": "Tipaza", "43": "Mila",
    "44": "Aïn Defla", "45": "Naâma", "46": "Aïn Témouchent",
    "47": "Ghardaïa", "48": "Relizane",
    "49": "El M'Ghair", "50": "El Menia",
    "51": "Ouled Djellal", "52": "Bordj Badji Mokhtar",
    "53": "Béni Abbès", "54": "Timimoun",
    "55": "Touggourt", "56": "Djanet",
    "57": "In Salah", "58": "In Guezzam"
}

# Expressions régulières
TIME_PATTERN = re.compile(r'^\d{2}:\d{2}-\d{2}:\d{2}$')
PLATE_PATTERN = re.compile(r'^\d{3,4}[A-Za-z]{2,3}\d{2,3}$')  # Exemple basique

def is_algerian_plate(text):
    """Vérifie si le texte correspond à une plaque algérienne"""
    digits_only = ''.join(c for c in text if c.isdigit())
    if len(digits_only) not in [9, 10, 11]:  # autoriser plus long
        return False
    wilaya_code = digits_only[-2:]
    return wilaya_code.isdigit() and 1 <= int(wilaya_code) <= 58

def classify_plate(text):
    """Classification complète du numéro de plaque algérienne"""
    try:
        # Nettoyer le texte et s'assurer que c'est une plaque algérienne
        clean_text = ''.join(c for c in text if c.isalnum()).upper()

        if len(clean_text) < 7 or not is_algerian_plate(clean_text):
            return None

        matricule_complet = clean_text
        position = clean_text[:-5]
        middle = clean_text[-5:-2]
        wilaya_code = clean_text[-2:]

        if not middle.isdigit() or not wilaya_code.isdigit():
            return None

        categorie = middle[0]
        annee = f"20{middle[1:]}" if middle[1:].isdigit() else "Inconnue"
        wilaya = WILAYAS.get(wilaya_code, "Wilaya inconnue")
        vehicle_type = CATEGORIES.get(categorie, "Catégorie inconnue")

        return {
            'matricule_complet': matricule_complet,
            'wilaya': (wilaya_code, wilaya),
            'annee': annee,
            'categorie': (categorie, vehicle_type),
            'serie': position
        }
    except Exception as e:
        print(f"Erreur de classification: {str(e)}")
        return None

def classify_plate_number(shared_results):
    """Classifier le numéro de plaque détecté uniquement si elle est algérienne"""
    # Check if we have text to classify
    if not shared_results.get("trocr_combined_text"):
        return "Aucun texte de plaque à classifier", "", "❌ Aucune plaque détectée", ""

    text = shared_results["trocr_combined_text"]

    if not is_algerian_plate(text):
        return "Plaque non algérienne détectée", "Type non détecté", "❌ Non algérienne", ""

    classified_plate = classify_plate(text)
    if not classified_plate:
        return "Impossible de classifier la plaque", "Type non détecté", "❌ Plaque invalide", ""

    # Prepare the classification result string
    classification_text = f"Plaque: {classified_plate['matricule_complet']}\n"
    classification_text += f"Wilaya: {classified_plate['wilaya'][1]} ({classified_plate['wilaya'][0]})\n"
    classification_text += f"Année: {classified_plate['annee']}\n"
    classification_text += f"Catégorie: {classified_plate['categorie'][1]} ({classified_plate['categorie'][0]})\n"
    classification_text += f"Série: {classified_plate['serie']}\n"

    # Save results if additional vehicle info is available
    if all(key in shared_results for key in ["label_color", "vehicle_model", "label_orientation"]):
        save_complete_results(
            plate_info=classified_plate,
            color=shared_results["label_color"],
            model=shared_results["vehicle_model"],
            orientation=shared_results["label_orientation"],
            vehicle_type=classified_plate['categorie'][1],
            brand=shared_results.get("vehicle_brand", "")
        )

    return (
        classification_text,
        f"Type: {classified_plate['categorie'][1]}" if classified_plate['categorie'][1] else "Type non détecté",
        "✅ Plaque algérienne",
        "Classification réussie"
    )

def save_complete_results(plate_info, color, model, orientation, vehicle_type, brand, filename="resultats.txt"):
    """Sauvegarde toutes les informations dans un fichier"""
    with open(filename, "a", encoding="utf-8") as f:
        f.write("\n" + "="*60 + "\n")
        f.write(f"ANALYSE EFFECTUÉE LE : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")

        # Section plaque d'immatriculation
        f.write("INFORMATIONS PLAQUE:\n")
        f.write("-"*50 + "\n")
        if plate_info:
            f.write(f"Numéro complet: {plate_info.get('matricule_complet', 'N/A')}\n")
            f.write(f"Wilaya: {plate_info.get('wilaya', ('', 'N/A'))[1]} ({plate_info.get('wilaya', ('', ''))[0]})\n")
            f.write(f"Année: {plate_info.get('annee', 'N/A')}\n")
            f.write(f"Catégorie: {plate_info.get('categorie', ('', 'N/A'))[1]} ({plate_info.get('categorie', ('', ''))[0]})\n")
            f.write(f"Série: {plate_info.get('serie', 'N/A')}\n")
        else:
            f.write("Aucune information de plaque disponible\n")

        # Section caractéristiques véhicule
        f.write("\nCARACTÉRISTIQUES VÉHICULE:\n")
        f.write("-"*50 + "\n")
        f.write(f"Couleur: {color if color else 'Non détectée'}\n")
        f.write(f"Marque: {brand if brand else 'Non détectée'}\n")
        f.write(f"Modèle: {model if model else 'Non détecté'}\n")
        f.write(f"Orientation: {orientation if orientation else 'Non détectée'}\n")
        f.write(f"Type de véhicule: {vehicle_type if vehicle_type else 'Non détecté'}\n")
        f.write("\n" + "="*60 + "\n\n")

def is_empty_plate(cropped_plate_image):
    """Détecte si la plaque est visuellement vide (espace blanc)"""
    if cropped_plate_image is None:
        return True

    if isinstance(cropped_plate_image, Image.Image):
        plate_img = np.array(cropped_plate_image)
    else:
        plate_img = cropped_plate_image

    gray = cv2.cvtColor(plate_img, cv2.COLOR_RGB2GRAY)
    _, thresholded = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    non_white_pixels = cv2.countNonZero(thresholded)
    total_pixels = gray.shape[0] * gray.shape[1]
    return non_white_pixels < (0.01 * total_pixels)

def draw_detection_boxes(image, shared_results):
    """Dessiner toutes les boîtes de détection sur l'image"""
    img_draw = image.copy() if isinstance(image, np.ndarray) else np.array(image.copy())

    # Convert to BGR if needed (OpenCV uses BGR format)
    if len(img_draw.shape) == 3 and img_draw.shape[2] == 3:
        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)

    # Boîte pour le véhicule
    if shared_results.get("vehicle_detected", False) and shared_results.get("vehicle_box"):
        x1, y1, x2, y2 = shared_results["vehicle_box"]
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(img_draw, "VEHICLE", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

    # Boîte pour la plaque
    if "plate" in shared_results.get("detection_boxes", {}):
        x1, y1, x2, y2 = shared_results["detection_boxes"]["plate"]
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_draw, "PLATE", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Convert back to RGB for display
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_draw)

    # Boîte pour le logo
    if shared_results["detection_boxes"]["logo"]:
        x1, y1, x2, y2 = shared_results["detection_boxes"]["logo"]
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Bleu pour logo
        cv2.putText(img_draw, "LOGO", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Ajouter le modèle si détecté
        if shared_results["vehicle_model"]:
            model_text = shared_results["vehicle_model"].split("(")[0].strip() if "(" in shared_results["vehicle_model"] else shared_results["vehicle_model"]
            cv2.putText(img_draw, f"Model: {model_text}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Boîte pour la couleur
    if shared_results["detection_boxes"]["color"]:
        x1, y1, x2, y2 = shared_results["detection_boxes"]["color"]
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Rouge pour couleur
        cv2.putText(img_draw, "COLOR", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Ajouter la couleur détectée
        if shared_results["label_color"]:
            cv2.putText(img_draw, f"{shared_results['label_color']}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Boîte pour l'orientation
    if shared_results["detection_boxes"]["orientation"]:
        x1, y1, x2, y2 = shared_results["detection_boxes"]["orientation"]
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Cyan pour orientation
        cv2.putText(img_draw, "ORIENTATION", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        # Ajouter l'orientation détectée
        if shared_results["label_orientation"]:
            cv2.putText(img_draw, f"{shared_results['label_orientation']}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return img_draw
