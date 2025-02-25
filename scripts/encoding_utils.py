import pandas as pd
import numpy as np
from category_encoders import HashingEncoder, OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

def encode_categorical_features(df, method="onehot", ordinal_cols=None, threshold=10, hashing_components=16):
    """
    Applique un encodage des variables catégorielles :
    - Ordinal Encoding pour les colonnes spécifiées dans `ordinal_cols`
    - "onehot" : One-Hot Encoding pour les autres colonnes catégorielles
    - "hashing" : Hashing Encoding pour les autres colonnes catégorielles
    
    Args :
    df : DataFrame contenant les données à transformer
    method : Méthode d'encodage pour les variables non ordinales, "onehot" ou "hashing"
    ordinal_cols : Liste des colonnes ordinales à traiter spécifiquement
    threshold : Seuil pour définir une faible ou forte cardinalité, non utilisé ici mais laissé pour flexibilité
    hashing_components : Nombre de dimensions pour l'encodage par hachage, par défaut 16
    
    Returns :
    DataFrame avec les colonnes catégorielles encodées
    """
    
    # Identification des colonnes catégorielles
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Détection des colonnes ordinales et exclusion des autres colonnes catégorielles
    ordinal_cols = ordinal_cols or []
    non_ordinal_cols = [col for col in categorical_cols if col not in ordinal_cols]

    # Vérification de la méthode d'encodage choisie
    valid_methods = ["onehot", "hashing"]
    if method not in valid_methods:
        raise ValueError(f"Méthode inconnue : {method}. Choisissez parmi {valid_methods}.")

    # Copie du DataFrame original
    df_encoded = df.copy()

    # Encodage Ordinal si des colonnes ordinales sont spécifiées
    if ordinal_cols:
        ord_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df_encoded[ordinal_cols] = ord_encoder.fit_transform(df_encoded[ordinal_cols])
    
    # Encodage des autres colonnes catégorielles avec la méthode choisie
    if non_ordinal_cols:
        if method == "onehot":
            encoder = OneHotEncoder(cols=non_ordinal_cols, use_cat_names=True, handle_unknown='ignore')
        elif method == "hashing":
            encoder = HashingEncoder(cols=non_ordinal_cols, n_components=hashing_components)
        
        df_encoded = encoder.fit_transform(df_encoded)

    # Affichage du résumé des encodages appliqués
    print("Résumé de l'encodage appliqué :")
    if ordinal_cols:
        print(f"- Ordinal Encoding : {ordinal_cols}")
    if non_ordinal_cols:
        print(f"- {method.capitalize()} Encoding : {non_ordinal_cols}")

    return df_encoded
