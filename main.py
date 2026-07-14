from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import re
from xgboost import XGBRegressor

#==========================================
# 1. FONCTION DE CHARGEMENT ET NETTOYAGE DU DATASET

def charger_et_nettoyer_dataset(chemin_fichier):
    df_brut = pd.read_csv(chemin_fichier, on_bad_lines='skip')
    df_brut.columns = df_brut.columns.str.strip()
    
    df_clean = pd.DataFrame()
    
    # Extraction de la marque
    def trouver_marque(nom):
        nom_str = str(nom)
        if nom_str.startswith("G.Skill"): return "G.Skill"
        if nom_str.startswith("Silicon Power"): return "Silicon Power"
        return nom_str.split()[0] if nom_str.split() else "Unknown"
        
    df_clean['brand'] = df_brut['Name'].apply(trouver_marque)
    
    # Extraction génération et fréquence
    generation_speed = df_brut['Speed'].str.split('-', n=1).str
    df_clean['generation'] = generation_speed[0]
    df_clean['frequency_mhz'] = pd.to_numeric(generation_speed[1], errors='coerce').fillna(2133).astype(int)
    
    # Extraction capacité
    def trouver_capacite(nom):
        match = re.search(r'(\d+)\s*GB', str(nom))
        return int(match.group(1)) if match else 16
        
    df_clean['capacity_gb'] = df_brut['Name'].apply(trouver_capacite)
    
    # Nettoyage et conversion du prix
    df_clean['price'] = df_brut['Price'].astype(str).str.replace('$', '', regex=False).str.strip()
    df_clean['price'] = pd.to_numeric(df_clean['price'], errors='coerce').fillna(0.0)
    df_clean['price_per_gb'] = df_brut['Price Per GB'].astype(str).str.replace('$', '', regex=False).str.strip()
    df_clean['price_per_gb'] = pd.to_numeric(df_clean['price_per_gb'], errors='coerce').fillna(0.0)
    
    # Création de la colonne numérique pour l'IA
    df_clean['generation_num'] = df_clean['generation'].map({'DDR4': 4, 'DDR5': 5}).fillna(4).astype(int)
    
    return df_clean

# ==========================================
# 2. CHARGEMENT ET FILTRAGE DU DATASET

df = charger_et_nettoyer_dataset('ram_dataset.csv')
df = df[df['price_per_gb'] > 0]

# L'IA apprend à deviner le PRIX PAR GO (pas le prix total)
X = df[['capacity_gb', 'frequency_mhz', 'generation_num']]
y = df['price_per_gb'] # <-- CHANGEMENT ICI

modele_ram = XGBRegressor(n_estimators=200, learning_rate=0.01, max_depth=3)
modele_ram.fit(X, y)

# ==========================================
# 4. CONFIGURATION DE L'API FASTAPI

app = FastAPI(title="RAM Pricing API", description="API to analyze RAM prices using AI", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequeteRam(BaseModel):
    capacity_gb: int
    frequency_mhz: int
    generation: str
    prix_vendeur: float

@app.post("/analyser-ram")
def analyser_ram(data: RequeteRam):
    generation_texte = data.generation.upper()
    
    # 🛑 SÉCURITÉ 1 : Anti-triche DDR4 (Fréquence trop haute)
    if generation_texte == 'DDR4' and data.frequency_mhz > 4400 or data.frequency_mhz < 2133:
        return {
            "prix_estime_ia": 0.00,
            "prix_vendeur_fourni": data.prix_vendeur,
            "ecart_pourcentage": 0.0,
            "verdict": "❌ Impossible configuration: DDR4 can't run at this frequency (2133 MHz minimum / 4400 MHz maximum)!",
            "couleur": "red"
        }

    # 🛑 SÉCURITÉ 2 : Anti-triche DDR5 (Fréquence trop basse)
    if generation_texte == 'DDR5' and data.frequency_mhz < 4800 or data.frequency_mhz > 8400:
        return {
            "prix_estime_ia": 0.00,
            "prix_vendeur_fourni": data.prix_vendeur,
            "ecart_pourcentage": 0.0,
            "verdict": "❌ Impossible configuration: DDR5 can't run at this frequency (minimum 4800 MHz / maximum 8400 MHz)!",
            "couleur": "red"
        }

    mapping_generation = {'DDR4': 4, 'DDR5': 5}
    gen_num = mapping_generation.get(generation_texte, 4)

    donnees_entree = pd.DataFrame([{
        'capacity_gb': data.capacity_gb,
        'frequency_mhz': data.frequency_mhz,
        'generation_num': gen_num
    }])
    
    # 1. L'IA prédit le prix d'UN Go
    prix_par_gb_estime = float(modele_ram.predict(donnees_entree)[0])
    
    # 2. Mathématiques : On multiplie par la capacité demandée pour avoir le prix total
    prix_ia = prix_par_gb_estime * data.capacity_gb # <-- MAGIE ICI
    
    # 3. Le reste du calcul d'écart ne change pas
    ecart = (prix_ia - data.prix_vendeur) / prix_ia
    
    # Logique des if/else pour le verdict...
    if ecart >= 0.40:
        statut = "🔥 Excellent deal !"
        couleur = "green"
    elif ecart <= -0.30:
        statut = "❌ Way too expensive."
        couleur = "red"
    else:
        statut = "⚖️ Price within the market standard."
        couleur = "blue"
        
    return {
        "prix_estime_ia": round(prix_ia, 2),
        "prix_vendeur_fourni": data.prix_vendeur,
        "ecart_pourcentage": round(ecart * 100, 1),
        "verdict": statut,
        "couleur": couleur
    }