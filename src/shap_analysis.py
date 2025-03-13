import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Activer SHAP JS pour un affichage interactif
shap.initjs()

def load_data(relative_path: str) -> pd.DataFrame:
    """
    Charge un fichier CSV en DataFrame en vérifiant son existence.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, relative_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ Le fichier {full_path} est introuvable.")

    print(f"📂 Chargement des données depuis : {full_path}")
    df = pd.read_csv(full_path)
    print(f"✅ Données chargées ({df.shape[0]} lignes, {df.shape[1]} colonnes).")
    return df

def reindex_features(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Assure que les features correspondent au modèle.
    """
    if hasattr(model, "feature_names_in_"):
        training_features = list(model.feature_names_in_)
        print(f"🔄 Reindexation des features ({len(training_features)} features)...")
        X = X.reindex(columns=training_features, fill_value=0)
    else:
        print("⚠️ Le modèle ne contient pas 'feature_names_in_'")
    return X

def load_latest_models(models_dir: str) -> dict:
    """
    Charge les modèles les plus récents depuis le dossier.
    """
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"❌ Dossier de modèles introuvable : {models_dir}")

    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
    if not model_files:
        raise FileNotFoundError("❌ Aucun modèle trouvé dans le dossier !")

    loaded_models = {}
    for model_name in ["RandomForest", "XGBoost", "LightGBM"]:
        latest_model = sorted(
            [f for f in model_files if f.startswith(model_name)], reverse=True
        )
        if latest_model:
            model_path = os.path.join(models_dir, latest_model[0])
            loaded_models[model_name] = joblib.load(model_path)
            print(f"✅ {model_name} chargé depuis {model_path}")
        else:
            print(f"⚠️ Aucun modèle {model_name} trouvé.")

    return loaded_models

def create_shap_plots(model, X_test, model_name, output_dir):
    """
    Génère et affiche les graphiques SHAP.
    """
    print(f"📊 Génération des explications SHAP pour {model_name}...")

    try:
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test)

        # Création du dossier si inexistant
        os.makedirs(output_dir, exist_ok=True)

        # SHAP Summary Plot
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        summary_path = os.path.join(output_dir, f"{model_name}_shap_summary.png")
        plt.savefig(summary_path, bbox_inches="tight")
        plt.show()
        print(f"📸 SHAP summary plot sauvegardé : {summary_path}")

        # SHAP Feature Importance Plot
        plt.figure(figsize=(8, 6))
        shap.plots.bar(shap_values, show=False)
        bar_path = os.path.join(output_dir, f"{model_name}_shap_bar.png")
        plt.savefig(bar_path, bbox_inches="tight")
        plt.show()
        print(f"📸 SHAP bar plot sauvegardé : {bar_path}")

    except Exception as e:
        print(f"❌ Erreur lors de la génération SHAP pour {model_name}: {e}")

def main():
    # Chemin des données test
    test_data_path = os.path.join("..", "data", "processed", "bmt_test.csv")
    df_test = load_data(test_data_path)

    # Vérification de la colonne cible
    if "survival_status" not in df_test.columns:
        raise ValueError("❌ Colonne cible 'survival_status' manquante !")

    # Préparation des features
    X_test = df_test.drop(columns=["survival_status"])
    X_test = pd.get_dummies(X_test, drop_first=True)  # Encodage catégoriel
    print(f"🔍 Données test transformées : {X_test.shape}")

    # Chargement des modèles
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "..", "models")
    models = load_latest_models(models_dir)

    if not models:
        return  # Arrêt si aucun modèle n'est chargé

    # Dossier de sortie pour SHAP
    shap_dir = os.path.join(script_dir, "..", "shap_analysis")
    os.makedirs(shap_dir, exist_ok=True)

    # Génération des analyses SHAP pour chaque modèle
    for model_name, model in models.items():
        print(f"\n🔍 Analyse SHAP pour {model_name}...")
        X_test_reindexed = reindex_features(X_test, model)
        create_shap_plots(model, X_test_reindexed, model_name, shap_dir)

if __name__ == "__main__":
    main()
