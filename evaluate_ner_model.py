from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None

from ner_project_utils import (
    LABEL_NAMES,
    build_confusion,
    evaluate_model,
    load_checkpoint,
    load_splits,
    make_dataloader,
    metrics_to_dict,
)


def peak_memory_mb() -> float | None:
    if resource is None:
        return None

    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB, macOS reports bytes.
    if max_rss > 10**7:
        return max_rss / (1024 * 1024)
    return max_rss / 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved NER checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to the saved .pt model checkpoint.")
    parser.add_argument("--data-dir", default="data/conll2003", help="Directory containing train.txt, valid.txt, and test.txt.")
    parser.add_argument("--split", choices=["validation", "test"], default="test", help="Which split to evaluate on.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size used during evaluation.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    parser.add_argument(
        "--output-json",
        default="evaluation_results.json",
        help="Where to save the machine-readable evaluation summary.",
    )
    parser.add_argument(
        "--save-confusion",
        default="confusion_matrix.npy",
        help="Where to save the normalized token-level confusion matrix as a NumPy array.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    start_time = time.perf_counter()
    model, vocab, metadata = load_checkpoint(args.checkpoint, device)
    splits = load_splits(args.data_dir)
    dataloader = make_dataloader(
        splits[args.split],
        vocab=vocab,
        batch_size=args.batch_size,
        shuffle=False,
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    results = evaluate_model(model, dataloader, criterion, device, LABEL_NAMES)
    elapsed_seconds = time.perf_counter() - start_time
    confusion = build_confusion(results.token_gold_ids, results.token_pred_ids, normalize="true")

    output = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "device": str(device),
        "model_config": metadata["config"],
        "saved_training_metrics": metadata.get("metrics", {}),
        "verification_time_seconds": elapsed_seconds,
        "peak_memory_mb": peak_memory_mb(),
        "num_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "metrics": metrics_to_dict(results),
    }

    Path(args.output_json).write_text(json.dumps(output, indent=2))
    np.save(args.save_confusion, confusion)

    print(f"Checkpoint: {output['checkpoint']}")
    print(f"Split: {args.split}")
    print(f"Device: {device}")
    print(f"Verification time (seconds): {elapsed_seconds:.4f}")
    if output["peak_memory_mb"] is not None:
        print(f"Peak memory (MB): {output['peak_memory_mb']:.2f}")
    print(f"Number of parameters: {output['num_parameters']}")
    print(f"Loss: {results.loss:.4f}")
    print(f"Token accuracy: {results.token_accuracy:.4f}")
    print(f"Entity precision: {results.entity_precision:.4f}")
    print(f"Entity recall: {results.entity_recall:.4f}")
    print(f"Entity F1: {results.entity_f1:.4f}")
    print("\nEntity-level classification report:")
    print(results.report_text)
    print(f"\nSaved JSON summary to: {Path(args.output_json).resolve()}")
    print(f"Saved token-level confusion matrix to: {Path(args.save_confusion).resolve()}")


if __name__ == "__main__":
    main()
