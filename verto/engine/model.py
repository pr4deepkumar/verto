"""
engine/model.py
~~~~~~~~~~~~~~~
T5-large + LoRA model loading and SQL generation.
"""

import logging
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
from transformers import T5ForConditionalGeneration, T5Tokenizer

logger = logging.getLogger(__name__)

# Suppress noisy torchao warnings about optional GPU extensions.
logging.getLogger("torchao").setLevel(logging.ERROR)

BASE_MODEL_NAME = "t5-large"

# Settings for a new LoRA adapter. Checkpoints written by scripts/train.py store the
# settings they were trained with, so a checkpoint with another rank still loads.
LORA_DEFAULTS = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.1, "target_modules": ["q", "v"]}

# T5's vocabulary has no "<", so it could neither read nor write comparisons like
# "age < 30". Text going into the model spells "<" as "&lt;", and generated SQL is
# turned back.
_T5_LESS_THAN = "&lt;"


def to_t5_text(text: str) -> str:
    return text.replace("<", _T5_LESS_THAN)


def from_t5_text(text: str) -> str:
    return text.replace(_T5_LESS_THAN, "<")


def default_device() -> torch.device:
    """CUDA if available, then Apple silicon's GPU (MPS), then the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lora_config(**settings) -> LoraConfig:
    """A seq2seq LoRA config: LORA_DEFAULTS with *settings* applied on top."""
    return LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, **{**LORA_DEFAULTS, **settings})


def load_model(
    checkpoint_path: str | Path | None = None,
    device: torch.device | str | None = None,
) -> tuple:
    """Load T5-large with the fine-tuned LoRA adapter merged into its weights.

    Args:
        checkpoint_path: Path to ``best_model.pt``. If ``None`` or the file doesn't
                         exist, the base model is returned without an adapter, which
                         is what an untrained adapter amounts to (useful for testing
                         the scaffold without a checkpoint).
        device: Target device. Defaults to CUDA, then MPS, then CPU.

    Returns:
        ``(model, tokenizer)`` — model is in eval mode and moved to *device*.
    """
    device = torch.device(device) if device is not None else default_device()

    logger.info("Loading tokeniser from %s …", BASE_MODEL_NAME)
    tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL_NAME)

    logger.info("Loading base model from %s …", BASE_MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            logger.info("Loading fine-tuned weights from %s …", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if "adapter" in checkpoint:  # written by scripts/train.py, with its LoRA settings
                model = get_peft_model(model, lora_config(**checkpoint["lora"]))
                set_peft_model_state_dict(model, checkpoint["adapter"])
            elif checkpoint and all("lora_" in k for k in checkpoint):  # older adapter-only files
                model = get_peft_model(model, lora_config())
                set_peft_model_state_dict(model, checkpoint)
            else:  # older full state dicts
                model = get_peft_model(model, lora_config())
                model.load_state_dict(checkpoint)
            # Fold the adapter into the base weights, so generation does no extra work per token.
            model = model.merge_and_unload()
        else:
            logger.warning(
                "Checkpoint not found at %s — using the base model without a fine-tuned adapter.",
                checkpoint_path,
            )

    model.to(device)
    model.eval()
    logger.info("Model ready on %s.", device)
    return model, tokenizer


def generate_candidates(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_length: int = 256,
    num_beams: int = 5,
) -> list[str]:
    """Run beam-search decoding and return every beam's SQL, best first.

    The first candidate is what plain beam search returns. The others cost nothing
    extra, since beam search computes them anyway; error recovery runs them when the
    first one fails.

    Args:
        model: The T5 model (in eval mode).
        tokenizer: Corresponding T5 tokenizer.
        prompt: Full input string including the schema.
        device: Device the model lives on.
        max_length: Maximum number of output tokens.
        num_beams: Beam width for beam-search decoding, and the number of candidates.

    Returns:
        Decoded SQL strings (special tokens stripped), highest-scoring first.
    """
    input_ids = (
        tokenizer(to_t5_text(prompt), return_tensors="pt", truncation=True, max_length=1024)
        .input_ids.to(device)
    )
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            early_stopping=True,
        )
    return [from_t5_text(tokenizer.decode(output, skip_special_tokens=True)) for output in outputs]
