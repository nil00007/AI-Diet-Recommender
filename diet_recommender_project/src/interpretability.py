import joblib, shap, matplotlib.pyplot as plt
from src.data_preprocessing import load_raw, build_pipe
from src.feature_engineering import add_feats
from src.config import MODEL_PATH, TARGET

saved = joblib.load(MODEL_PATH)
model = saved["model"]

df = add_feats(load_raw())
X = df.drop(columns=[TARGET])
explainer = shap.Explainer(model.named_steps["clf"])
shap_values = explainer(model.named_steps["pre"].transform(X))
shap.summary_plot(shap_values, show=False)
plt.tight_layout()
plt.savefig("models/shap_summary.png")
