"""
scripts/train.py
~~~~~~~~~~~~~~~~
Standalone training script for the VERTO NLDI Engine.

Usage
-----
    python scripts/train.py --spider-dir data/spider
    python scripts/train.py --epochs 5 --lr 1e-3 --batch-size 8 --grad-accum 2
    python scripts/train.py --checkpoint-dir runs/exp1 --no-eval
    python scripts/train.py --eval-only

All arguments have sensible defaults; run with --help for the full list.
The best checkpoint is saved to <checkpoint-dir>/best_model.pt, and the evaluation on the
Spider dev set to results/spider_dev_eval.json.
"""

from __future__ import annotations

# ── Path bootstrap (allows running without `pip install -e .`) ────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import gc
import json
import logging
import platform
import random
import statistics
import textwrap
import time
from collections import Counter
from datetime import datetime, timezone

import peft
import torch
import transformers
from peft import get_peft_model, get_peft_model_state_dict
from torch.amp import autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)

from verto.engine.executor import execute_sql, normalize_sql
from verto.engine.inference import NLDIEngine, build_prompt
from verto.engine.model import BASE_MODEL_NAME, LORA_DEFAULTS, default_device, lora_config, to_t5_text
from verto.engine.schema import load_and_format_schemas

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Dataset ───────────────────────────────────────────────────────────────────

class SQLDataset(Dataset):
    """Spider examples as (prompt, target SQL) pairs, spelled the way the model reads them."""

    def __init__(self, examples: list[dict], schema_dict: dict[str, str]) -> None:
        self.pairs = [
            (to_t5_text(build_prompt(ex["question"], schema_dict.get(ex["db_id"], ""))), to_t5_text(ex["query"]))
            for ex in examples
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.pairs[idx]


def make_collate(tokenizer, max_input_length: int = 1024, max_target_length: int = 256):
    """Tokenise a batch, padded to its own longest prompt rather than to max_input_length."""

    def collate(batch: list[tuple[str, str]]) -> dict:
        inputs, targets = zip(*batch, strict=True)
        input_enc = tokenizer(list(inputs), max_length=max_input_length, truncation=True,
                              padding="longest", return_tensors="pt")
        target_enc = tokenizer(list(targets), max_length=max_target_length, truncation=True,
                               padding="longest", return_tensors="pt")
        labels = target_enc.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        return {"input_ids": input_enc.input_ids, "attention_mask": input_enc.attention_mask, "labels": labels}

    return collate


class LengthBuckets(Sampler):
    """Batches of prompts with similar lengths, in random order.

    Shuffles the examples, sorts each run of 50 batches by prompt length, cuts it into
    batches and shuffles the batches, so each batch pads to little more than its real
    length. On an Apple M4 this took an epoch from about 4.5 hours to 48 minutes.
    """

    def __init__(self, lengths: list[int], batch_size: int, seed: int) -> None:
        self.lengths, self.batch_size, self.rng = lengths, batch_size, random.Random(seed)

    def __iter__(self):
        order = list(range(len(self.lengths)))
        self.rng.shuffle(order)
        chunk = self.batch_size * 50
        batches = []
        for start in range(0, len(order), chunk):
            run = sorted(order[start:start + chunk], key=self.lengths.__getitem__)
            batches += [run[i:i + self.batch_size] for i in range(0, len(run), self.batch_size)]
        self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def make_loader(dataset: SQLDataset, tokenizer, batch_size: int, seed: int) -> DataLoader:
    lengths = [len(tokenizer(prompt).input_ids) for prompt, _ in dataset.pairs]
    return DataLoader(dataset, batch_sampler=LengthBuckets(lengths, batch_size, seed),
                      collate_fn=make_collate(tokenizer))


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(lora_settings: dict) -> tuple:
    log.info("Loading %s tokeniser …", BASE_MODEL_NAME)
    tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL_NAME)

    log.info("Loading %s base model …", BASE_MODEL_NAME)
    base_model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)

    model = get_peft_model(base_model, lora_config(**lora_settings))
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    return model, tokenizer


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(
    checkpoint_dir: Path, model, lora_settings: dict, optimizer, scheduler,
    epoch: int, val_loss: float, is_best: bool = False,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"epoch": epoch, "val_loss": val_loss,
         "optimizer": optimizer.state_dict(),
         "scheduler": scheduler.state_dict()},
        checkpoint_dir / "latest_checkpoint.pt",
    )
    if is_best:
        # Save ONLY the LoRA adapter weights (a few MB) rather than the full ~3 GB T5-large
        # state dict, with the LoRA settings the server needs to rebuild it.
        torch.save({"lora": lora_settings, "adapter": get_peft_model_state_dict(model)},
                   checkpoint_dir / "best_model.pt")
        log.info("  ✓ New best model saved (val_loss=%.4f)", val_loss)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    model,
    train_loader: DataLoader, val_loader: DataLoader,
    optimizer, scheduler,
    device: torch.device, num_epochs: int,
    gradient_accumulation_steps: int, checkpoint_dir: Path, lora_settings: dict,
) -> None:
    best_val_loss = float("inf")
    trainable = [p for p in model.parameters() if p.requires_grad]
    # bfloat16 on Apple silicon and on Ampere or newer GPUs. Older GPUs (Colab's T4) and the
    # CPU train in float32: T5 overflows in float16, and bfloat16 is slow without hardware support.
    use_bf16 = device.type == "mps" or (device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8)

    for epoch in range(1, num_epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        started = time.time()

        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss    = outputs.loss / gradient_accumulation_steps

            loss.backward()
            total_loss += outputs.loss.item()

            if step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 100 == 0:
                log.info("  epoch %d  step %d/%d  loss %.4f  %.2fs/step", epoch, step,
                         len(train_loader), total_loss / step, (time.time() - started) / step)

        avg_train = total_loss / len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    out = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device),
                    )
                val_loss += out.loss.item()

        avg_val = val_loss / len(val_loader)
        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val

        log.info(
            "Epoch %02d/%02d  train_loss=%.4f  val_loss=%.4f  %.1f min%s",
            epoch, num_epochs, avg_train, avg_val, (time.time() - started) / 60, "  ← best" if is_best else "",
        )
        save_checkpoint(checkpoint_dir, model, lora_settings, optimizer, scheduler, epoch, avg_val, is_best)


def run_training(args: argparse.Namespace, spider_dir: Path, checkpoint_dir: Path,
                 device: torch.device, lora_settings: dict) -> float:
    """Train an adapter and save the best one to checkpoint_dir. Returns the minutes it took."""
    schema_dict = load_and_format_schemas(spider_dir / "tables.json")
    log.info("Schemas loaded for %d databases.", len(schema_dict))

    with open(spider_dir / "train_spider.json") as f:
        examples = json.load(f)
    random.Random(args.seed).shuffle(examples)
    n_val = int(len(examples) * args.val_split)
    val_data, train_data = examples[:n_val], examples[n_val:]
    log.info("Dataset: %d train / %d val", len(train_data), len(val_data))

    model, tokenizer = build_model(lora_settings)
    model.to(device)

    train_loader = make_loader(SQLDataset(train_data, schema_dict), tokenizer, args.batch_size, args.seed)
    val_loader   = make_loader(SQLDataset(val_data, schema_dict), tokenizer, args.batch_size, args.seed + 1)

    n_steps   = (len(train_loader) // args.grad_accum) * args.epochs
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                    num_warmup_steps=int(0.1 * n_steps), num_training_steps=n_steps)

    log.info("Training: epochs=%d  batch=%d×%d  lr=%s  device=%s",
             args.epochs, args.batch_size, args.grad_accum, args.lr, device)

    started = time.time()
    train(model, train_loader, val_loader, optimizer, scheduler, device,
          args.epochs, args.grad_accum, checkpoint_dir, lora_settings)
    return (time.time() - started) / 60


# ── Evaluation ────────────────────────────────────────────────────────────────

def same_rows(gold_sql: str, gold_rows: list[tuple], pred_rows: list[tuple]) -> bool:
    """Execution match: the same rows, in the same order when the gold query sorts them."""
    if "order by" in gold_sql.lower():
        return pred_rows == gold_rows
    return Counter(pred_rows) == Counter(gold_rows)


def evaluate(engine: NLDIEngine, examples: list[dict], use_schema_linking: bool = True) -> dict:
    """Ask every question through engine.query(), the path the server uses, and score it.

    A first-attempt failure is a question whose top candidate failed to run; "after retry"
    means one of the model's other candidates ran instead.
    """
    exact = exec_match = errors = gold_errors = 0
    first_fail = ran_after_retry = right_after_retry = 0
    seconds, token_cut = [], []
    total = len(examples)
    log.info("Evaluating on %d examples …", total)

    for i, ex in enumerate(examples):
        if i % 100 == 0:
            log.info("  %d / %d", i, total)
        question, db_id, gold_sql = ex["question"], ex["db_id"], ex["query"]

        started = time.perf_counter()
        out = engine.query(question, db_id, use_schema_linking=use_schema_linking)
        seconds.append(time.perf_counter() - started)
        gold = execute_sql(gold_sql, engine.db_dir / db_id / f"{db_id}.sqlite")

        if out["sql"] and normalize_sql(out["sql"]) == normalize_sql(gold_sql):
            exact += 1
        right = False
        if out["error"] is not None:
            errors += 1
        elif isinstance(gold, str):
            gold_errors += 1  # the gold SQL itself fails, so this one can't be scored
        else:
            right = same_rows(gold_sql, gold[1], [tuple(row) for row in out["result"]])
            exec_match += right
        if out["attempts"] > 1:
            first_fail += 1
            if out["error"] is None:
                ran_after_retry += 1
                right_after_retry += right

        full = engine.tokenizer(build_prompt(question, engine.schema_for(question, db_id, False))).input_ids
        linked = engine.tokenizer(build_prompt(question, engine.schema_for(question, db_id, True))).input_ids
        token_cut.append(1 - len(linked) / len(full))

    pct = lambda n: round(n / total * 100, 1)
    seconds.sort()
    return {
        "questions":                       total,
        "execution_accuracy":              pct(exec_match),
        "exact_match":                     pct(exact),
        "sql_error_rate":                  pct(errors),
        "first_attempt_failures":          first_fail,
        "ran_after_retry":                 ran_after_retry,
        "right_after_retry":               right_after_retry,
        "gold_sql_errors":                 gold_errors,
        "seconds_per_question_median":     round(statistics.median(seconds), 2),
        "seconds_per_question_p90":        round(seconds[int(0.9 * (total - 1))], 2),
        "schema_linking_token_cut_median": round(statistics.median(token_cut) * 100, 1),
        "schema_linking_token_cut_mean":   round(statistics.mean(token_cut) * 100, 1),
    }


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return f"Apple GPU (MPS), {platform.machine()}"
    return f"CPU, {platform.machine()}"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python scripts/train.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
            Train the VERTO NLDI Engine (T5-large + LoRA) on the Spider dataset.

            Examples
            --------
              python scripts/train.py
              python scripts/train.py --epochs 5 --lr 1e-3 --batch-size 8 --grad-accum 2
              python scripts/train.py --eval-only
        """),
    )
    p.add_argument("--spider-dir",     default="data/spider",  metavar="PATH")
    p.add_argument("--checkpoint-dir", default="checkpoints",  metavar="PATH")
    p.add_argument("--eval-output",    default="results/spider_dev_eval.json", metavar="PATH",
                   help="Where to write the evaluation results as JSON.")
    p.add_argument("--epochs",         type=int,   default=20)
    p.add_argument("--batch-size",     type=int,   default=4)
    p.add_argument("--grad-accum",     type=int,   default=8)
    p.add_argument("--lr",             type=float, default=5e-5)
    p.add_argument("--lora-rank",      type=int,   default=LORA_DEFAULTS["r"])
    p.add_argument("--lora-alpha",     type=int,   default=LORA_DEFAULTS["lora_alpha"])
    p.add_argument("--val-split",      type=float, default=0.1)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         default=None, help="cuda, mps or cpu [default: auto]")
    p.add_argument("--no-schema-linking", action="store_true",
                   help="Evaluate with the full schema in every prompt.")
    p.add_argument("--no-eval",        action="store_true")
    p.add_argument("--eval-only",      action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    spider_dir     = Path(args.spider_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    tables_json    = spider_dir / "tables.json"

    if not tables_json.exists():
        raise FileNotFoundError(
            f"tables.json not found at {tables_json}\n"
            "Run:  python scripts/setup_data.py"
        )

    device = torch.device(args.device) if args.device else default_device()
    log.info("Device: %s", device)
    best_ckpt = checkpoint_dir / "best_model.pt"
    lora_settings = {**LORA_DEFAULTS, "r": args.lora_rank, "lora_alpha": args.lora_alpha}

    train_minutes = None
    if not args.eval_only:
        train_minutes = run_training(args, spider_dir, checkpoint_dir, device, lora_settings)
        gc.collect()  # free the training model before the engine loads its own copy
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    if args.no_eval:
        return
    if not best_ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {best_ckpt}")

    engine = NLDIEngine(checkpoint_path=best_ckpt, spider_dir=spider_dir, device=str(device))
    with open(spider_dir / "dev.json") as f:
        dev = json.load(f)
    started = time.time()
    results = evaluate(engine, dev, use_schema_linking=not args.no_schema_linking)
    results["eval_minutes"] = round((time.time() - started) / 60, 1)
    results["run"] = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "device": device_name(device),
        "schema_linking": not args.no_schema_linking,
        "error_recovery": "next beam-search candidates",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "training": None if args.eval_only else {
            "epochs": args.epochs, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "lr": args.lr, "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
            "seed": args.seed, "minutes": round(train_minutes, 1),
        },
    }
    out_path = Path(args.eval_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")

    print()
    print("=" * 64)
    print(f"  Spider dev evaluation ({results['questions']:,} questions, {results['run']['device']})")
    print("=" * 64)
    print(f"  Execution accuracy    : {results['execution_accuracy']}%")
    print(f"  Exact match           : {results['exact_match']}%")
    print(f"  SQL error rate        : {results['sql_error_rate']}%")
    print(f"  Failed first attempt  : {results['first_attempt_failures']}, of which "
          f"{results['ran_after_retry']} ran with another candidate ({results['right_after_retry']} right)")
    print(f"  Seconds per question  : median {results['seconds_per_question_median']}, "
          f"p90 {results['seconds_per_question_p90']}")
    print(f"  Schema linking        : {'on' if not args.no_schema_linking else 'off'} (when on, prompts "
          f"{results['schema_linking_token_cut_median']}% shorter, median)")
    print(f"  Results saved to      : {out_path}")
    print("=" * 64)
    print()


if __name__ == "__main__":
    main()
