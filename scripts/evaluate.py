# scripts/evaluate.py
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import yaml
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.data import load_dataset


def load_cfg(path="config/train.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_cfg()
    art_dir = Path(cfg.get("artifacts_dir", "artifacts"))

    model = joblib.load(art_dir / "model.joblib")
    X, y = load_dataset(cfg)

    # Évaluation sur le MÊME jeu de test que train.py (jamais sur les données
    # d'entraînement, sinon le score est artificiellement optimiste).
    _, Xte, _, yte = train_test_split(
        X, y,
        test_size=float(cfg["split"]["test_size"]),
        random_state=int(cfg["split"]["random_state"]),
        stratify=y,
    )
    pred = model.predict(Xte)

    report = classification_report(yte, pred, output_dict=True)
    report["accuracy_test"] = float(accuracy_score(yte, pred))
    (art_dir / "report.json").write_text(json.dumps(report, indent=2))
    print("Evaluate OK: artifacts/report.json")
    print("Accuracy (test):", report["accuracy_test"])


if __name__ == "__main__":
    main()
