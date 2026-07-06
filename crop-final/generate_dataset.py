import numpy as np
import pandas as pd

np.random.seed(7)

CROPS = ["Wheat","Rice","Maize","Cotton","Sugarcane","Soybean","Groundnut","Mustard","Chickpea","Tomato"]
DISTRICTS = ["Indore","Bhopal","Ujjain","Dewas","Ratlam","Nagpur","Pune","Ludhiana","Meerut","Coimbatore"]
SEASONS = ["Kharif","Rabi","Zaid"]
N_SAMPLES = 5000

CROP_PROFILES = {
    "Wheat":     dict(ideal_ph=6.8,rain_sens=0.6,humid_sens=0.5,temp_sens=0.4),
    "Rice":      dict(ideal_ph=6.0,rain_sens=0.3,humid_sens=0.9,temp_sens=0.5),
    "Maize":     dict(ideal_ph=6.5,rain_sens=0.7,humid_sens=0.6,temp_sens=0.6),
    "Cotton":    dict(ideal_ph=7.5,rain_sens=0.5,humid_sens=0.7,temp_sens=0.7),
    "Sugarcane": dict(ideal_ph=6.5,rain_sens=0.4,humid_sens=0.8,temp_sens=0.5),
    "Soybean":   dict(ideal_ph=6.5,rain_sens=0.6,humid_sens=0.8,temp_sens=0.5),
    "Groundnut": dict(ideal_ph=6.2,rain_sens=0.5,humid_sens=0.7,temp_sens=0.6),
    "Mustard":   dict(ideal_ph=7.0,rain_sens=0.3,humid_sens=0.4,temp_sens=0.7),
    "Chickpea":  dict(ideal_ph=7.2,rain_sens=0.4,humid_sens=0.5,temp_sens=0.6),
    "Tomato":    dict(ideal_ph=6.3,rain_sens=0.7,humid_sens=0.9,temp_sens=0.6),
}

def compute_risk(crop,soil_ph,rainfall_mm,temperature_c,humidity_pct,nitrogen,phosphorus,potassium):
    p = CROP_PROFILES[crop]
    ph_pen = abs(soil_ph - p["ideal_ph"]) * 8
    hum_eff = (humidity_pct - 50) * p["humid_sens"] * 0.6
    rain_eff = max(0, rainfall_mm - 800) * p["rain_sens"] * 0.02
    temp_eff = max(0, temperature_c - 30) * p["temp_sens"] * 1.2
    nut_pen = max(0,40-nitrogen)*0.15 + max(0,20-phosphorus)*0.2 + max(0,20-potassium)*0.15
    score = 25 + ph_pen + hum_eff + rain_eff + temp_eff + nut_pen + np.random.normal(0,0.5)
    return float(np.clip(score,0,100))

def label(score):
    if score < 36: return "Low"
    elif score < 52: return "Medium"
    else: return "High"

rows = []
for _ in range(N_SAMPLES):
    crop = np.random.choice(CROPS)
    district = np.random.choice(DISTRICTS)
    season = np.random.choice(SEASONS)
    soil_ph = float(np.clip(np.random.normal(6.7,0.7),4.5,9.0))
    rainfall_mm = float(np.clip(np.random.normal(900,350),50,2500))
    temperature_c = float(np.clip(np.random.normal(28,5),10,45))
    humidity_pct = float(np.clip(np.random.normal(60,18),15,100))
    nitrogen = float(np.clip(np.random.normal(45,15),0,100))
    phosphorus = float(np.clip(np.random.normal(25,10),0,80))
    potassium = float(np.clip(np.random.normal(25,10),0,80))
    soil_moisture_pct = float(np.clip(np.random.normal(35,12),5,90))
    score = compute_risk(crop,soil_ph,rainfall_mm,temperature_c,humidity_pct,nitrogen,phosphorus,potassium)
    rows.append({"crop":crop,"district":district,"season":season,"soil_ph":round(soil_ph,2),"rainfall_mm":round(rainfall_mm,1),"temperature_c":round(temperature_c,1),"humidity_pct":round(humidity_pct,1),"nitrogen":round(nitrogen,1),"phosphorus":round(phosphorus,1),"potassium":round(potassium,1),"soil_moisture_pct":round(soil_moisture_pct,1),"disease_risk":label(score)})

df = pd.DataFrame(rows)
df.to_csv("crop_disease_data.csv",index=False)
print(f"Saved {len(df)} rows")
print(df["disease_risk"].value_counts())
