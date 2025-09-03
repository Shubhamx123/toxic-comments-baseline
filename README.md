# toxic-comments-baseline

Baseline NLP pipeline for **toxic comment** detection using **TF–IDF + Logistic Regression**.  
Clean, reproducible, and easy to extend.

## Quickstart
```bash
# 1) (optional) create a venv
python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) install deps
pip install -r requirements.txt

# 3) train on the sample data
python src/train.py --data data/sample.csv --outdir results

# 4) predict on new text (stdin)
echo "I hate you!" | python src/predict.py --model results/model.joblib --vectorizer results/vectorizer.joblib
```
Outputs (metrics + artifacts) appear in `results/`.

## Structure
```
.
├── data/
│   └── sample.csv
├── src/
│   ├── train.py
│   └── predict.py
├── tests/
│   └── test_smoke.py
├── .github/workflows/ci.yml
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Dataset
Use `data/sample.csv` to smoke-test. For real training, replace with a larger dataset (e.g., Jigsaw Toxic Comments).  
CSV columns: **text,label** where `label ∈ {0,1}`.

## Results (sample run)
Toy set will report very high metrics due to tiny size. On a real dataset, expect more realistic scores.

---
© 2025 Shubham Raj — MIT