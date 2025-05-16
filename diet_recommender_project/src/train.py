import numpy as np, joblib, warnings
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_raw, build_pipe
from src.feature_engineering import add_feats
from src.config import MODEL_PATH, TARGET, SEED
warnings.filterwarnings("ignore", category=FutureWarning)

NOISE_RATE = 0.10        # %10 label-noise hem train hem test

def main():
    np.random.seed(SEED)

    # ── Load & features
    df = add_feats(load_raw())

    # ── Encode labels
    le = LabelEncoder()
    df["y_num"] = le.fit_transform(df[TARGET].str.strip())
    class_map = dict(zip(le.transform(le.classes_), le.classes_))

    # ── Inject noise BEFORE split (affects train & test)
    noise_idx = df.sample(frac=NOISE_RATE, random_state=SEED).index
    df.loc[noise_idx, "y_num"] = np.random.randint(0, 3, size=len(noise_idx))

    # ── Split
    X = df.drop(columns=[TARGET, "y_num"])
    y = df["y_num"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    # Sample weights
    freq = y_tr.value_counts(normalize=True)
    w = y_tr.map(lambda c: 1 / freq[c])

    pre = build_pipe()

    model = Pipeline([
        ("pre", pre),
        ("clf", XGBClassifier(
            random_state=SEED, eval_metric="mlogloss",
            n_estimators=1200, max_depth=5, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9))
    ])

    model.fit(X_tr, y_tr, clf__sample_weight=w)

    f1 = f1_score(y_te, model.predict(X_te), average="macro")
    print(f"XGBoost macro-F1: {f1:.3f}")

    joblib.dump({"model": model, "map": class_map}, MODEL_PATH)
    print("Saved →", MODEL_PATH)

if __name__ == "__main__":
    main()
