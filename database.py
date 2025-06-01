import sqlite3
import re
from datetime import datetime

DB_PATH = "models/vehicules_database.db"
TIME_PATTERN = re.compile(r'^\d{2}:\d{2}-\d{2}:\d{2}$')

def create_connection():
    """Créer une connexion à la base SQLite"""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        return conn
    except Error as e:
        print(f"Erreur de connexion à SQLite: {e}")
    return conn

def init_database():
    """Initialiser la base de données SQLite"""
    conn = create_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vehicules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plaque TEXT NOT NULL UNIQUE,
                    marque TEXT,
                    modele TEXT,
                    couleur TEXT,
                    statut TEXT,
                    plage_horaire TEXT,
                    date_enregistrement TEXT
                )
            """)
            conn.commit()
        except Error as e:
            print(f"Erreur création table: {e}")
        finally:
            conn.close()

def check_vehicle(plate_text):
    """Vérifier si un véhicule existe dans la base"""
    init_database()
    conn = create_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT statut, plage_horaire FROM vehicules WHERE plaque = ?", (plate_text,))
            result = cursor.fetchone()

            if result:
                return True, f"Statut: {result[0]} | Accès: {result[1]}"
            return False, "Véhicule non enregistré"
        except Error as e:
            print(f"Erreur lecture base: {e}")
            return False, "Erreur base de données"
        finally:
            conn.close()
    return False, "Erreur de connexion"

def save_vehicle(plate_info, color, model, brand, status, time_range):
    """Enregistrer un nouveau véhicule"""
    init_database()
    conn = create_connection()
    if conn is not None:
        try:
            # Nettoyer les données
            plate_number = str(plate_info['matricule_complet']).strip()
            clean_brand = brand.split('(')[0].strip() if '(' in brand else brand
            clean_model = model.split('(')[0].strip() if '(' in model else model

            cursor = conn.cursor()

            # Vérifier si le véhicule existe déjà
            cursor.execute("SELECT 1 FROM vehicules WHERE plaque = ?", (plate_number,))
            if cursor.fetchone():
                return False, "Véhicule déjà existant"

            # Insérer le nouveau véhicule
            cursor.execute("""
                INSERT INTO vehicules (plaque, marque, modele, couleur, statut, plage_horaire, date_enregistrement)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                plate_number,
                clean_brand,
                clean_model,
                color,
                status,
                time_range,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))

            conn.commit()
            return True, "Enregistrement réussi"
        except Error as e:
            return False, f"Erreur enregistrement: {e}"
        finally:
            conn.close()
    return False, "Erreur de connexion"

def is_access_allowed(plate_text):
    """Vérifier si l'accès est autorisé à l'heure actuelle"""
    conn = create_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT statut, plage_horaire FROM vehicules WHERE plaque = ?", (plate_text,))
            vehicle = cursor.fetchone()

            if not vehicle:
                return False

            if vehicle[0] == "Non Autorisé":
                return False

            if vehicle[1] == "24/24":
                return True

            current_time = datetime.now().time()
            start_str, end_str = vehicle[1].split('-')
            start = time(*map(int, start_str.split(':')))
            end = time(*map(int, end_str.split(':')))

            return start <= current_time <= end
        except Error as e:
            print(f"Erreur vérification accès: {e}")
            return False
        finally:
            conn.close()
    return False

# Autres fonctions de gestion de la base de données...
