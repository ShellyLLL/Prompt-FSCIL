"""Auditable session-wise FSCIL accuracy and AA/PD calculations."""
import csv
import json
import os

import numpy as np
import torch


def _as_long_tensor(values):
    return torch.as_tensor(values, dtype=torch.long).reshape(-1).cpu()


def class_accuracy_map(labels, predictions, class_ids):
    labels = _as_long_tensor(labels)
    predictions = _as_long_tensor(predictions)
    if labels.numel() != predictions.numel() or labels.numel() == 0:
        raise ValueError("Labels and predictions must be non-empty and aligned.")
    result = {}
    for class_id in class_ids:
        class_id = int(class_id)
        mask = labels == class_id
        if not bool(mask.any()):
            raise RuntimeError(
                "Evaluation is missing expected class {}.".format(class_id)
            )
        result[class_id] = float((predictions[mask] == labels[mask]).float().mean())
    return result


def mean_class_accuracy(labels, predictions, class_ids):
    values = class_accuracy_map(labels, predictions, class_ids)
    return float(np.mean(list(values.values())))


def evaluate_session_predictions(
    labels, predictions, base_class, seen_classes, session, way
):
    """Return macro class accuracies, sample accuracy and a full confusion matrix."""
    labels = _as_long_tensor(labels)
    predictions = _as_long_tensor(predictions)
    base_class = int(base_class)
    seen_classes = int(seen_classes)
    session = int(session)
    way = int(way)
    if labels.numel() != predictions.numel() or labels.numel() == 0:
        raise ValueError("Evaluation labels and predictions must be non-empty.")
    if session < 0 or base_class < 1 or seen_classes < base_class or way < 1:
        raise ValueError("Invalid session/class configuration.")
    expected = list(range(seen_classes))
    per_class = class_accuracy_map(labels, predictions, expected)

    current_start = base_class + (session - 1) * way if session > 0 else base_class
    if current_start > seen_classes:
        raise ValueError("Current-session class range exceeds seen classes.")
    old_seen_ids = list(range(current_start))
    base_ids = list(range(base_class))
    old_novel_ids = list(range(base_class, current_start))
    current_new_ids = list(range(current_start, seen_classes))

    def group_accuracy(ids):
        if not ids:
            return None
        missing = [class_id for class_id in ids if class_id not in per_class]
        if missing:
            raise RuntimeError("Missing classes in accuracy group: {}".format(missing))
        return float(np.mean([per_class[class_id] for class_id in ids]))

    confusion = torch.bincount(
        labels * seen_classes + predictions.clamp(min=0, max=seen_classes - 1),
        minlength=seen_classes * seen_classes,
    ).reshape(seen_classes, seen_classes)

    if torch.any(labels < 0) or torch.any(labels >= seen_classes):
        raise RuntimeError("Evaluation labels fall outside the seen class range.")
    if torch.any(predictions < 0) or torch.any(predictions >= seen_classes):
        raise RuntimeError("Predictions fall outside the seen class range.")

    return {
        "session": session,
        "seen_classes": seen_classes,
        "evaluation_samples": int(labels.numel()),
        "sample_accuracy": float((labels == predictions).float().mean()),
        "class_average_accuracy": float(np.mean(list(per_class.values()))),
        "base_class_average_accuracy": group_accuracy(base_ids),
        "old_seen_class_average_accuracy": group_accuracy(old_seen_ids),
        "old_novel_class_average_accuracy": group_accuracy(old_novel_ids),
        "current_new_class_average_accuracy": group_accuracy(current_new_ids),
        "per_class_accuracy": {
            str(class_id): per_class[class_id] for class_id in expected
        },
        "confusion_matrix": confusion.tolist(),
    }


def summarize_session_metrics(session_metrics):
    """AA is the mean of session macro accuracies; PD is A0 minus AT."""
    records = sorted(session_metrics, key=lambda item: int(item["session"]))
    if not records:
        raise ValueError("At least one evaluated session is required.")
    expected_sessions = list(range(len(records)))
    actual_sessions = [int(item["session"]) for item in records]
    if actual_sessions != expected_sessions:
        raise RuntimeError(
            "Session metrics must contain exactly S0..ST; got {}".format(
                actual_sessions
            )
        )
    accuracies = [float(item["class_average_accuracy"]) for item in records]
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in accuracies):
        raise ValueError("Session class-average accuracies must be finite fractions.")
    aa = float(np.mean(accuracies))
    pd = float(accuracies[0] - accuracies[-1])
    return {
        "AA": aa,
        "PD": pd,
        "AA_percent": 100.0 * aa,
        "PD_percentage_points": 100.0 * pd,
        "base_session_accuracy": accuracies[0],
        "final_session_accuracy": accuracies[-1],
        "session_class_average_accuracy": accuracies,
        "session_class_average_accuracy_percent": [
            100.0 * value for value in accuracies
        ],
        "session_sample_accuracy": [
            float(item["sample_accuracy"]) for item in records
        ],
        "num_sessions": len(records),
    }


def write_session_csv(path, session_metrics):
    fields = [
        "session", "seen_classes", "evaluation_samples",
        "base_class_average_accuracy",
        "old_seen_class_average_accuracy",
        "old_novel_class_average_accuracy",
        "current_new_class_average_accuracy",
        "class_average_accuracy", "sample_accuracy",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in sorted(session_metrics, key=lambda row: int(row["session"])):
            row = {key: item.get(key) for key in fields}
            for key in fields[3:]:
                value = row[key]
                row[key] = "" if value is None else "{:.4f}".format(100.0 * float(value))
            writer.writerow(row)


def write_confusion_csv(path, confusion):
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Confusion matrix must be square.")
    np.savetxt(path, matrix, delimiter=",", fmt="%d")


def format_session_table(session_metrics):
    rows = [
        "Session | Seen | Base | Old novel | Current new | All-seen macro | Sample Top-1",
        "-------:|-----:|-----:|----------:|------------:|---------------:|------------:",
    ]
    for item in sorted(session_metrics, key=lambda row: int(row["session"])):
        def pct(key):
            value = item.get(key)
            return "--" if value is None else "{:.2f}%".format(100.0 * float(value))
        rows.append(
            "S{session} | {seen} | {base} | {old_novel} | {new} | {macro} | {sample}".format(
                session=int(item["session"]),
                seen=int(item["seen_classes"]),
                base=pct("base_class_average_accuracy"),
                old_novel=pct("old_novel_class_average_accuracy"),
                new=pct("current_new_class_average_accuracy"),
                macro=pct("class_average_accuracy"),
                sample=pct("sample_accuracy"),
            )
        )
    summary = summarize_session_metrics(session_metrics)
    rows.append("")
    rows.append(
        "AA = {:.2f}% (mean of S0..ST macro accuracies)".format(
            summary["AA_percent"]
        )
    )
    rows.append(
        "PD = {:.2f} percentage points (S0 macro - ST macro)".format(
            summary["PD_percentage_points"]
        )
    )
    return "\n".join(rows)
