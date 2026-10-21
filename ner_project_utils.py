from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

LABEL_NAMES = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_LABEL_ID = -100


def parse_conll(path: str | Path) -> list[dict[str, list[str]]]:
    examples: list[dict[str, list[str]]] = []
    tokens: list[str] = []
    ner_tags: list[str] = []

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("-DOCSTART-") or not line.strip():
                if tokens:
                    examples.append({"tokens": tokens, "ner_tags": ner_tags})
                    tokens, ner_tags = [], []
                continue

            splits = line.rstrip().split(" ")
            tokens.append(splits[0])
            ner_tags.append(splits[3])

    if tokens:
        examples.append({"tokens": tokens, "ner_tags": ner_tags})

    return examples


def load_splits(data_dir: str | Path) -> dict[str, list[dict[str, list[str]]]]:
    data_path = Path(data_dir)
    return {
        "train": parse_conll(data_path / "train.txt"),
        "validation": parse_conll(data_path / "valid.txt"),
        "test": parse_conll(data_path / "test.txt"),
    }


def build_vocab(examples: list[dict[str, list[str]]], min_freq: int = 1) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        for token in example["tokens"]:
            lowered = token.lower()
            counts[lowered] = counts.get(lowered, 0) + 1

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in sorted(counts.items()):
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


class NERTokenDataset(Dataset):
    def __init__(self, examples: list[dict[str, list[str]]], vocab: dict[str, int], label_names: list[str]):
        self.examples = examples
        self.vocab = vocab
        self.label_to_id = {label: index for index, label in enumerate(label_names)}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        token_ids = [self.vocab.get(token.lower(), self.vocab[UNK_TOKEN]) for token in example["tokens"]]
        label_ids = [self.label_to_id[tag] for tag in example["ner_tags"]]
        return {
            "tokens": example["tokens"],
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    tokens = [item["tokens"] for item in batch]

    padded_inputs = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=PAD_LABEL_ID)
    attention_mask = padded_inputs.ne(0).long()

    return {
        "tokens": tokens,
        "input_ids": padded_inputs,
        "labels": padded_labels,
        "attention_mask": attention_mask,
    }


def make_dataloader(
    examples: list[dict[str, list[str]]],
    vocab: dict[str, int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = NERTokenDataset(examples, vocab, LABEL_NAMES)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)


class FlexibleNERTagger(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_dim: int = 16,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        hidden_dims = hidden_dims or []
        layers: list[nn.Module] = []
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.encoder = nn.Sequential(*layers) if layers else nn.Identity()
        self.classifier = nn.Linear(input_dim, num_labels)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        encoded = self.encoder(embeddings)
        return self.classifier(encoded)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


@dataclass
class EvalResults:
    loss: float
    token_accuracy: float
    entity_precision: float
    entity_recall: float
    entity_f1: float
    token_gold_ids: list[int]
    token_pred_ids: list[int]
    entity_gold_labels: list[list[str]]
    entity_pred_labels: list[list[str]]
    report_text: str


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    label_names: list[str] | None = None,
) -> EvalResults:
    label_names = label_names or LABEL_NAMES
    model.eval()

    total_loss = 0.0
    token_gold_ids: list[int] = []
    token_pred_ids: list[int] = []
    entity_gold_labels: list[list[str]] = []
    entity_pred_labels: list[list[str]] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

            predictions = logits.argmax(dim=-1).cpu().numpy()
            gold_labels = labels.cpu().numpy()

            for predicted_row, gold_row in zip(predictions, gold_labels):
                sentence_gold: list[str] = []
                sentence_pred: list[str] = []
                for predicted_id, gold_id in zip(predicted_row, gold_row):
                    if int(gold_id) == PAD_LABEL_ID:
                        continue
                    token_gold_ids.append(int(gold_id))
                    token_pred_ids.append(int(predicted_id))
                    sentence_gold.append(label_names[int(gold_id)])
                    sentence_pred.append(label_names[int(predicted_id)])
                entity_gold_labels.append(sentence_gold)
                entity_pred_labels.append(sentence_pred)

    gold_array = np.array(token_gold_ids)
    pred_array = np.array(token_pred_ids)
    token_accuracy = float((gold_array == pred_array).mean()) if len(gold_array) else 0.0

    return EvalResults(
        loss=total_loss / len(dataloader),
        token_accuracy=token_accuracy,
        entity_precision=precision_score(entity_gold_labels, entity_pred_labels),
        entity_recall=recall_score(entity_gold_labels, entity_pred_labels),
        entity_f1=f1_score(entity_gold_labels, entity_pred_labels),
        token_gold_ids=token_gold_ids,
        token_pred_ids=token_pred_ids,
        entity_gold_labels=entity_gold_labels,
        entity_pred_labels=entity_pred_labels,
        report_text=classification_report(
            entity_gold_labels,
            entity_pred_labels,
            digits=4,
            zero_division=0,
        ),
    )


def build_confusion(token_gold_ids: list[int], token_pred_ids: list[int], normalize: str | None = "true") -> np.ndarray:
    return confusion_matrix(
        token_gold_ids,
        token_pred_ids,
        labels=list(range(len(LABEL_NAMES))),
        normalize=normalize,
    )


def checkpoint_payload(
    model: nn.Module,
    vocab: dict[str, int],
    config: dict[str, Any],
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "label_names": LABEL_NAMES,
        "config": config,
        "metrics": extra_metrics or {},
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    vocab: dict[str, int],
    config: dict[str, Any],
    extra_metrics: dict[str, Any] | None = None,
) -> None:
    torch.save(checkpoint_payload(model, vocab, config, extra_metrics), path)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, int], dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    vocab = checkpoint["vocab"]
    label_names = checkpoint.get("label_names", LABEL_NAMES)
    model = FlexibleNERTagger(
        vocab_size=len(vocab),
        num_labels=len(label_names),
        embedding_dim=config["embedding_dim"],
        hidden_dims=config.get("hidden_dims", []),
        dropout=config.get("dropout", 0.0),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    metadata = {
        "config": config,
        "label_names": label_names,
        "metrics": checkpoint.get("metrics", {}),
    }
    return model, vocab, metadata


def metrics_to_dict(results: EvalResults) -> dict[str, Any]:
    return {
        "loss": results.loss,
        "token_accuracy": results.token_accuracy,
        "entity_precision": results.entity_precision,
        "entity_recall": results.entity_recall,
        "entity_f1": results.entity_f1,
        "classification_report": results.report_text,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2))
