from pathlib import Path
import json, os, subprocess, sys

def test_train_and_artifacts(tmp_path: Path):
    # Copy sample data into tmpdir
    d = tmp_path / "data"
    d.mkdir()
    (d / "sample.csv").write_text("text,label\nhello,0\nI hate you,1\nawesome,0\n", encoding="utf-8")
    out = tmp_path / "results"
    # Run training via subprocess to simulate real CLI
    cmd = [sys.executable, "src/train.py", "--data", str(d / "sample.csv"), "--outdir", str(out)]
    subprocess.check_call(cmd, cwd=Path(__file__).resolve().parents[1])
    assert (out / "model.joblib").exists()
    assert (out / "vectorizer.joblib").exists()
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert "accuracy" in metrics