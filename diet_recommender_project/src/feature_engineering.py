import pandas as pd

def add_feats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── BMI ──
    if "BMI" not in df.columns:
        df["BMI"] = df["Weight_kg"] / ((df["Height_cm"] / 100) ** 2)

    # ── Disease grouping (avoid 1-to-1 leakage) ──
    df["Disease_Group"] = df["Disease_Type"].map({
        "Cardiac":       "Cardio",
        "Hypertension":  "Metabolic",
        "Diabetes":      "Metabolic",
        "None":          "Healthy"
    })

    # ── Lifestyle score ──
    lvl = {"Low": 0, "Medium": 1, "High": 2}
    df["Lifestyle_Score"] = (
        0.5 * df["Physical_Activity_Level"].str.strip().map(lvl).fillna(0) +
        0.3 * df["Weekly_Exercise_Hours"] / 10 +
        0.2 * df["Adherence_to_Diet_Plan"] / 10
    )

    # ── Numeric risk score (0-6) ──
    sev = {"Low": 0, "Moderate": 1, "High": 2}
    df["Risk_Score"] = (
        (df["BMI"] > 30).astype(int) +                 # obesity
        (df["Blood_Pressure_mmHg"] > 130).astype(int) +# high BP
        (df["Cholesterol_mg/dL"] > 200).astype(int) +  # high chol
        df["Severity"].map(sev).fillna(0)
    )

    # ── Binary medical flags ──
    df["Carb_Flag"] = (df["Glucose_mg/dL"] > 120).astype(int)
    df["Salt_Flag"] = (df["Blood_Pressure_mmHg"] > 130).astype(int)

    # ── Interaction: high glucose * high BMI ──
    df["Carb_Risk"] = df["Carb_Flag"] * (df["BMI"] > 28).astype(int)

    return df
