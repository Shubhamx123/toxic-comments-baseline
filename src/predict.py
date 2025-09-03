import argparse, sys, joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vectorizer", required=True)
    ap.add_argument("--file", default=None, help="Optional path to a text file; otherwise read stdin")
    args = ap.parse_args()

    clf = joblib.load(args.model)
    vec = joblib.load(args.vectorizer)

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = [line.strip() for line in sys.stdin if line.strip()]

    X = vec.transform(texts)
    preds = clf.predict(X)
    for t, p in zip(texts, preds):
        print(f"{p}\t{t}")

if __name__ == "__main__":
    main()