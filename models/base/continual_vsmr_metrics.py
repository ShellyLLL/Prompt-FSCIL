"""Auditable session-wise FSCIL accuracy and AA/PD calculations."""
import csv

import numpy as np
import torch


def _as_long_tensor(values):
    return torch.as_tensor(values, dtype=torch.long).reshape(-1).cpu()


def class_accuracy_map(labels, predictions, class_ids):
    labels = _as_long_tensor(labels)
    predictions = _as_long_tensor(predictions)
    class_ids = [int(value) for value in class_ids]
    if not class_ids or len(set(class_ids)) != len(class_ids):
        raise ValueError("class_ids must be a non-empty sequence of unique IDs.")
    if labels.numel() != predictions.numel() or labels.numel() == 0:
        raise ValueError("Labels and predictions must be non-empty and aligned.")
    expected = set(class_ids)
    observed_labels = set(labels.tolist())
    observed_predictions = set(predictions.tolist())
    if observed_labels != expected:
        raise RuntimeError(
            "Evaluation labels must contain exactly the expected classes; "
            "expected {}, observed {}.".format(sorted(expected), sorted(observed_labels))
        )
    invalid_predictions = observed_predictions - expected
    if invalid_predictions:
        raise RuntimeError(
            "Predictions contain class IDs outside the evaluated class set: {}."
            .format(sorted(invalid_predictions))
        )
    result = {}
    for class_id in class_ids:
        mask = labels == class_id
        result[class_id] = float(
            (predictions[mask] == labels[mask]).float().mean()
        )
    return result


def mean_class_accuracy(labels, predictions, class_ids):
    values = class_accuracy_map(labels, predictions, class_ids)
    return float(np.mean(list(values.values())))


def evaluate_session_predictions(
    labels, predictions, base_class, seen_classes, session, way
):
    """Evaluate exactly the classes seen at this Session.

    Class-average accuracy is the unweighted mean of per-class top-1 accuracy.
    Sample accuracy is reported separately and is never substituted for AA/PD.
    """
    labels = _as_long_tensor(labels)
    predictions = _as_long_tensor(predictions)
    base_class = int(base_class)
    seen_classes = int(seen_classes)
    session = int(session)
    way = int(way)
    if labels.numel() != predictions.numel() or labels.numel() == 0:
        raise ValueError("Evaluation labels and predictions must be non-empty.")
    if session < 0 or base_class < 1 or way < 1:
        raise ValueError("Invalid session/class configuration.")
    expected_seen = base_class + session * way
    if seen_classes != expected_seen:
        raise ValueError(
            "Session {} must evaluate exactly {} seen classes, got {}."
            .format(session, expected_seen, seen_classes)
        )

    expected = list(range(seen_classes))
    per_class = class_accuracy_map(labels, predictions, expected)

    current_start = base_class + (session - 1) * way if session > 0 else base_class
    old_seen_ids = list(range(current_start))
    base_ids = list(range(base_class))
    old_novel_ids = list(range(base_class, current_start))
    current_new_ids = list(range(current_start, seen_classes))

    def group_accuracy(ids):
        if not ids:
            return None
        return float(np.mean([per_class[class_id] for class_id in ids]))

    # Bounds and exact label coverage have already been checked above, so
    # bincount cannot silently clamp invalid predictions into the last class.
    encoded = labels * seen_classes + predictions
    confusion = torch.bincount(
        encoded, minlength=seen_classes * seen_classes
    ).reshape(seen_classes, seen_classes)

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
    """AA = mean S0..ST macro accuracy; PD = S0 macro accuracy - ST macro accuracy."""
    records = sorted(session_metrics, key=lambda item: int(item["session"]))
    if not records:
        raise ValueError("At least one evaluated session is required.")
    expected_sessions = list(range(len(records)))
    actual_sessions = [int(item["session"]) for item in records]
    if actual_sessions != expected_sessions:
        raise RuntimeError(
            "Session metrics must contain exactly S0..ST; got {}"
            .format(actual_sessions)
        )
    accuracies = [float(item["class_average_accuracy"]) for item in records]
    if not all(
        np.isfinite(value) and 0.0 <= value <= 1.0
        for value in accuracies
    ):
        raise ValueError(
            "Session class-average accuracies must be finite fractions in [0,1]."
        )
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
        "AA_definition": "Arithmetic mean of all-seen-class macro Top-1 accuracy over S0..ST.",
        "PD_definition": "S0 all-seen-class macro Top-1 accuracy minus ST all-seen-class macro Top-1 accuracy; reported in percentage points.",
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
                row[key] = (
                    "" if value is None
                    else "{:.4f}".format(100.0 * float(value))
                )
            writer.writerow(row)


def write_confusion_csv(path, confusion):
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Confusion matrix must be square.")
    np.savetxt(path, matrix, delimiter=",", fmt="%d")


def format_session_table(session_metrics):
    rows = [
        "Session | Seen | Base | Previously seen | Old novel | Current new | All-seen macro | Sample Top-1",
        "-------:|-----:|-----:|---------------:|----------:|------------:|---------------:|------------:",
    ]
    for item in sorted(session_metrics, key=lambda row: int(row["session"])):
        def pct(key):
            value = item.get(key)
            return "--" if value is None else "{:.2f}%".format(100.0 * float(value))
        rows.append(
            "S{session} | {seen} | {base} | {old_seen} | {old_novel} | {new} | {macro} | {sample}".format(
                session=int(item["session"]),
                seen=int(item["seen_classes"]),
                base=pct("base_class_average_accuracy"),
                old_seen=pct("old_seen_class_average_accuracy"),
                old_novel=pct("old_novel_class_average_accuracy"),
                new=pct("current_new_class_average_accuracy"),
                macro=pct("class_average_accuracy"),
                sample=pct("sample_accuracy"),
            )
        )
    summary = summarize_session_metrics(session_metrics)
    rows.append("")
    rows.append(
        "AA = {:.2f}% (arithmetic mean of S0..ST all-seen-class macro accuracy)"
        .format(summary["AA_percent"])
    )
    rows.append(
        "PD = {:.2f} percentage points (S0 macro accuracy - ST macro accuracy)"
        .format(summary["PD_percentage_points"])
    )
    rows.append(
        "Sample-weighted Top-1 is reported separately and is not used for AA/PD."
    )
    return "\n".join(rows)
