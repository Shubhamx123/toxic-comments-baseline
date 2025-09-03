import argparse, json, os, math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib


def _safe_split(texts, labels, base_test_size=0.2, seed=42):
    """
    Robust split for tiny/imbalanced datasets.

    - If dataset is very small (<5 rows) OR any class has <2 samples,
      we SKIP splitting and evaluate on the training set (nosplit mode).
    - Otherwise, try a stratified split; if test size would be too small,
      increase it so test has >= 1 sample per class.
    """
    n = len(labels)
    n_classes = labels.nunique()
    counts = labels.value_counts()
    min_count = int(counts.min()) if n_classes > 0 else 0

    # Nosplit for extreme small / singleton-class cases
    if n < 5 or n_classes < 2 or min_count < 2:
        return texts, texts, labels, labels, {
            "mode": "nosplit",
            "n": n,
            "n_classes": int(n_classes),
            "min_class_count": int(min_count),
        }

    # Otherwise, stratified split with a safe test size
    test_size = base_test_size
    n_test = math.ceil(n * test_size)
    if n_test < n_classes:
        # Ensure at least one sample per class in test (cap at 50%)
        min_frac = min(0.5, (n_classes / n) + 0.01)
        test_size = max(test_size, min_frac)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    return X_train, X_test, y_train, y_test, {
        "mode": "stratified",
        "n": n,
        "n_classes": int(n_classes),
        "min_class_count": int(min_count),
        "test_size": test_size,
    }


def train(data_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(data_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    texts = df["text"].astype(str)
    labels = df["label"].astype(int)

    X_train, X_test, y_train, y_test, split_info = _safe_split(
        texts, labels, base_test_size=0.2, seed=42
    )

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(max_iter=400)
    clf.fit(Xtr, y_train)

    preds = clf.predict(Xte)
    acc = accuracy_score(y_test, preds)

    # If binary-F1 is undefined (e.g., only one class present), fallback to macro F1.
    try:
        f1 = f1_score(y_test, preds, average="binary")
    except ValueError:
        f1 = f1_score(y_test, preds, average="macro")

    report = classification_report(y_test, preds, output_dict=True)

    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": split_info,
                "accuracy": acc,
                "f1": f1,
                "report": report,
                "notes": (
                    "Nosplit mode is used for very small/imbalanced datasets; "
                    "on real datasets, a stratified split is applied."
                ),
            },
            f,
            indent=2,
        )

    joblib.dump(clf, os.path.join(outdir, "model.joblib"))
    joblib.dump(vec, os.path.join(outdir, "vectorizer.joblib"))
    print(f"Saved artifacts to {outdir}")
    return acc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    train(args.data, args.outdir)


if __name__ == "__main__":
    main()
