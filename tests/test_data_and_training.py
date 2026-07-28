from __future__ import annotations

import csv
import os
import tempfile
import unittest

import torch
from torch import nn

from data.chest_xray14 import _stratified_val_split, load_chest_xray14
from training.trainer import TrainConfig, Trainer


class DatasetFallbackTests(unittest.TestCase):
    def test_missing_split_files_fall_back_to_csv_for_train_and_val(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            images = os.path.join(root, "images")
            os.makedirs(images)
            rows = [
                ("a.png", "Effusion"),
                ("b.png", "No Finding"),
                ("c.png", "Mass"),
                ("d.png", "No Finding"),
            ]
            with open(
                os.path.join(root, "Data_Entry_2017.csv"),
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["Image Index", "Finding Labels"]
                )
                writer.writeheader()
                for name, labels in rows:
                    writer.writerow(
                        {"Image Index": name, "Finding Labels": labels}
                    )
                    open(os.path.join(images, name), "wb").close()

            train = load_chest_xray14(root, split="train", val_fraction=0.5)
            val = load_chest_xray14(root, split="val", val_fraction=0.5)

            train_names = {os.path.basename(sample.path) for sample in train}
            val_names = {os.path.basename(sample.path) for sample in val}
            self.assertFalse(train_names & val_names)
            self.assertEqual(train_names | val_names, {name for name, _ in rows})

    def test_stratification_never_splits_one_patient_across_folds(self) -> None:
        names = {
            "00000001_000.png",
            "00000001_001.png",
            "00000002_000.png",
            "00000002_001.png",
            "00000003_000.png",
            "00000003_001.png",
        }
        vectors = {
            name: [1.0, 0.0] if name.startswith("00000001") else [0.0, 1.0]
            for name in names
        }
        validation = _stratified_val_split(
            names, vectors, n_labels=2, val_fraction=0.34, seed=42
        )

        for patient in ("00000001", "00000002", "00000003"):
            patient_names = {name for name in names if name.startswith(patient)}
            self.assertIn(
                patient_names & validation,
                (set(), patient_names),
                f"patient {patient} leaked across train and validation",
            )


class FakeEvaluator:
    class_names = ["negative", "positive"]

    def reset(self) -> None:
        pass


class RecordingTrainer(Trainer):
    def __init__(self) -> None:
        super().__init__(
            nn.Linear(1, 1),
            FakeEvaluator(),
            TrainConfig(epochs=1, mixed_precision=False, resume=False),
            device="cpu",
        )
        self.saved: list[tuple[str, float]] = []

    def _train_one_epoch(self, loader) -> float:
        return 0.1

    def _validate(self, loader) -> dict:
        return {"auc": 0.8, "val_loss": 0.2}

    def _save(self, name: str, epoch: int) -> None:
        self.saved.append((name, self.best_metric))


class TrainerCheckpointTests(unittest.TestCase):
    def test_last_checkpoint_records_updated_best_metric(self) -> None:
        trainer = RecordingTrainer()
        trainer.fit(None, None)
        self.assertEqual(
            trainer.saved,
            [("best.pt", 0.8), ("last.pt", 0.8)],
        )


if __name__ == "__main__":
    unittest.main()
