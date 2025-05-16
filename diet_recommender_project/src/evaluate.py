import joblib, pandas as pd
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)
from src.data_preprocessing import load_raw, split
from src.feature_engineering import add_feats
from src.config import MODEL_PATH, TARGET

def main():
    # ── load model
    saved = joblib.load(MODEL_PATH)
    model, class_map = saved["model"], saved["map"]

    # ── prepare test set
    df = add_feats(load_raw())
    _, X_te, _, y_te = split(df, test=0.20)

    # predictions → string labels
    y_pred = pd.Series(model.predict(X_te)).map(class_map)

    # ── detailed metrics
    print(classification_report(y_te, y_pred))

    # ── confusion matrix (console)
    cm = confusion_matrix(y_te, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{lbl}"  for lbl in class_map.values()],
        columns=[f"Pred_{lbl}" for lbl in class_map.values()]
    )
    print("\nConfusion Matrix:")
    print(cm_df.to_string())                  # neatly aligned matrix

    # ── overall accuracy
    print("\nAccuracy:", f"{accuracy_score(y_te, y_pred):.4f}")

if __name__ == "__main__":
    main()
