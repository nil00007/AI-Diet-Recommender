import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

CSV_PATH   = DATA_DIR / "diet_recommendations.csv"
MODEL_PATH = MODEL_DIR / "best_model.pkl"

TARGET = "Diet_Recommendation"

CAT_COLS = [
    "Gender", "Severity", "Disease_Type",  # ← geri eklendi
    "Disease_Group",
    "Dietary_Restrictions", "Allergies",
    "Preferred_Cuisine", "Physical_Activity_Level"
]

NUM_COLS = [
    "Age","Weight_kg","Height_cm","BMI",
    "Cholesterol_mg/dL","Blood_Pressure_mmHg","Glucose_mg/dL",
    "Weekly_Exercise_Hours","Daily_Caloric_Intake",
    "Adherence_to_Diet_Plan",
    "Lifestyle_Score","Risk_Score",
    "Carb_Flag","Salt_Flag","Carb_Risk"
]

SEED = 42
