import gradio as gr
import joblib, pandas as pd, warnings
from sklearn.metrics import accuracy_score, confusion_matrix
from src.feature_engineering import add_feats
from src.data_preprocessing import load_raw, split
from src.config import MODEL_PATH
from src.train import main as train_model
warnings.filterwarnings("ignore", category=FutureWarning)

# ── model load / train ─────────────────────────────────────
try:
    saved = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(">> Model file missing – training now …")
    train_model()
    saved = joblib.load(MODEL_PATH)

model, class_map = saved["model"], saved["map"]

# ── accuracy + confusion matrix (console) ─────────────────
df = add_feats(load_raw())
_, X_test, _, y_test = split(df, test=0.20)
y_pred = [class_map[c] for c in model.predict(X_test)]
acc    = accuracy_score(y_test, y_pred)

print(f"Model test accuracy : {acc:.4f}")
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", pd.DataFrame(
        cm,
        index=[f"T_{l}" for l in class_map.values()],
        columns=[f"P_{l}" for l in class_map.values()]
).to_string())

acc_banner = f"**Model test accuracy : {acc:.4f}**"

# ── recipe library (Breakfast / Lunch / Dinner) ───────────
RECIPES = {
    "Balanced": {
        "Breakfast": "Oatmeal with berries & Greek yogurt",
        "Lunch": "Grilled chicken + quinoa & mixed-greens salad",
        "Dinner": "Mediterranean veggie bowl with hummus & whole-grain pita"
    },
    "Low_Carb": {
        "Breakfast": "Cheese & spinach omelette, avocado slices",
        "Lunch": "Zucchini-noodle turkey Bolognese",
        "Dinner": "Salmon fillet with roasted broccoli & almond slivers"
    },
    "Low_Sodium": {
        "Breakfast": "Unsalted peanut-butter banana overnight oats",
        "Lunch": "No-salt lentil & vegetable soup",
        "Dinner": "Herb-roasted chicken, steamed sweet-potato mash"
    }
}

# ── field names ───────────────────────────────────────────
FIELD_NAMES = [
    "Age","Gender","Weight_kg","Height_cm","Disease_Type","Severity",
    "Cholesterol_mg/dL","Blood_Pressure_mmHg","Glucose_mg/dL",
    "Physical_Activity_Level","Weekly_Exercise_Hours",
    "Dietary_Restrictions","Allergies","Preferred_Cuisine",
    "Daily_Caloric_Intake","Adherence_to_Diet_Plan"
]

# ── predict (returns diet + nicely formatted recipes) ─────
def predict(*vals):
    inp = dict(zip(FIELD_NAMES, vals))
    if isinstance(inp["Allergies"], list):
        inp["Allergies"] = ",".join(sorted(inp["Allergies"])) or "None"

    diet = class_map[model.predict(add_feats(pd.DataFrame([inp])))[0]]
    r = RECIPES[diet]
    recipe_md = (
        f"### Breakfast\n- {r['Breakfast']}\n\n"
        f"### Lunch\n- {r['Lunch']}\n\n"
        f"### Dinner\n- {r['Dinner']}"
    )
    return diet, recipe_md

import gradio as gr
from gradio.components import Slider, Dropdown, Number, CheckboxGroup, Textbox, Markdown

# ── UI components ─────────────────────────────────────────
inputs = [
    Slider(18, 90, label="Age"),
    Dropdown(["Male", "Female"], label="Gender"),
    Number(label="Weight (kg)"),
    Number(label="Height (cm)"),
    Dropdown(["None", "Diabetes", "Hypertension", "Cardiac"], label="Disease Type"),
    Dropdown(["Low", "Moderate", "High"], label="Severity"),
    Number(label="Cholesterol (mg/dL)"),
    Number(label="Blood Pressure (mmHg)"),
    Number(label="Glucose (mg/dL)"),
    Dropdown(["Low", "Medium", "High"], label="Physical Activity Level"),
    Number(label="Weekly Exercise Hours"),
    Dropdown(["None", "Vegetarian", "Vegan", "Gluten_Free"], label="Dietary Restrictions"),
    CheckboxGroup(
        ["Dairy", "Eggs", "Fish", "Gluten", "Peanuts", "Shellfish", "Soy", "Tree_Nuts"],
        label="Allergies (select all that apply)"
    ),
    Dropdown(["Turkish", "Asian", "Mediterranean", "Other"], label="Preferred Cuisine"),
    Number(label="Daily Caloric Intake"),
    Slider(1, 10, label="Adherence to Diet Plan")
]

outputs = [
    Textbox(label="Recommended Diet Plan"),
    Markdown()
]


demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    title="Personalized Diet Recommender",
    description=acc_banner
)

if __name__ == "__main__":
    demo.launch()
