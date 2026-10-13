"""
validate_model.py
=================
Self-contained model architecture validation for Phase 2.

Runs 8 checks to verify the decoder-only Transformer is correctly implemented.
No training, no dataset, no GPU required.

Usage (from the repository root — RECOMMENDED)
-----------------------------------------------
    python -m nepali.model.validate_model

This avoids relative-import issues that can occur with:
    python nepali/model/validate_model.py

Both invocations are supported.

Checks performed
----------------
  [1] Config loading         — YAML loads and validates correctly.
  [2] Model instantiation    — DecoderOnlyTransformer can be created.
  [3] Architecture print     — Module tree is printed for inspection.
  [4] Parameter count        — Exact count printed; assert ~25 M.
  [5] Forward pass shape     — Logits are (B, T, vocab_size).
  [6] Weight tying           — Embedding and output weights share storage.
  [7] Causal masking         — Changing future token does not change past logits.
  [8] Context overflow       — ValueError raised when T > context_length.
"""

from __future__ import annotations

import os
import sys

# ─── Path setup ──────────────────────────────────────────────────────────────
# Support running as both:
#   python -m nepali.model.validate_model   (recommended, repo root)
#   python nepali/model/validate_model.py   (direct)
#
# When run directly, __package__ is None and relative imports fail.
# We detect this and insert the repo root (two directories up from this file).
if __package__ is None or __package__ == "":
    # This file is at: <repo_root>/nepali/model/validate_model.py
    # Add repo root to sys.path so absolute imports (nepali.model.*) work.
    _this_file = os.path.abspath(__file__)
    _model_dir = os.path.dirname(_this_file)         # .../nepali/model/
    _nepali_dir = os.path.dirname(_model_dir)         # .../nepali/
    _repo_root = os.path.dirname(_nepali_dir)         # .../individual-project-charan-1230/
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

import torch

# Relative imports work when run as a module (-m nepali.model.validate_model).
# Absolute imports work when the repo root is on sys.path (direct execution).
try:
    from .config import ModelConfig
    from .transformer import DecoderOnlyTransformer
except ImportError:
    from nepali.model.config import ModelConfig
    from nepali.model.transformer import DecoderOnlyTransformer


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _pass(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    raise AssertionError(msg)


# ─── Config path ─────────────────────────────────────────────────────────────

# The YAML path is resolved relative to the repo root.
# This file lives at: <repo_root>/nepali/model/validate_model.py
# So: two dirname() calls up → repo root.
_CONFIG_THIS_FILE = os.path.abspath(__file__)
_CONFIG_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_CONFIG_THIS_FILE)))
CONFIG_PATH = os.path.join(_CONFIG_REPO_ROOT, "nepali", "configs", "model_config.yaml")


# ─── Check 1: Config loading ──────────────────────────────────────────────────

def check_config_loading() -> ModelConfig:
    _section("Check 1 — Configuration loading")
    print(f"  Config path : {CONFIG_PATH}")

    cfg = ModelConfig.from_yaml(CONFIG_PATH)

    print(f"\n  Loaded config:\n{cfg}")

    # Spot-check a few values against the known architecture spec.
    assert cfg.vocab_size == 10_000,       f"Expected vocab_size=10000, got {cfg.vocab_size}"
    assert cfg.d_model == 384,             f"Expected d_model=384, got {cfg.d_model}"
    assert cfg.num_heads == 6,             f"Expected num_heads=6, got {cfg.num_heads}"
    assert cfg.num_layers == 12,           f"Expected num_layers=12, got {cfg.num_layers}"
    assert cfg.ffn_dim == 1536,            f"Expected ffn_dim=1536, got {cfg.ffn_dim}"
    assert cfg.context_length == 512,      f"Expected context_length=512, got {cfg.context_length}"
    assert cfg.head_dim == 64,             f"Expected head_dim=64, got {cfg.head_dim}"
    assert cfg.embedding_dropout == 0.1,   f"Expected embedding_dropout=0.1"
    assert cfg.attention_dropout == 0.1,   f"Expected attention_dropout=0.1"
    assert cfg.residual_dropout == 0.1,    f"Expected residual_dropout=0.1"

    _pass("Config loaded and validated correctly.")
    return cfg


# ─── Check 2: Model instantiation ────────────────────────────────────────────

def check_instantiation(cfg: ModelConfig) -> DecoderOnlyTransformer:
    _section("Check 2 — Model instantiation")

    model = DecoderOnlyTransformer(cfg)

    assert isinstance(model, DecoderOnlyTransformer), \
        "Expected a DecoderOnlyTransformer instance."

    _pass("DecoderOnlyTransformer instantiated successfully.")
    return model


# ─── Check 3: Architecture print ─────────────────────────────────────────────

def check_architecture_print(model: DecoderOnlyTransformer) -> None:
    _section("Check 3 — Architecture print")
    print(model)
    _pass("Architecture printed above.")


# ─── Check 4: Parameter count ─────────────────────────────────────────────────

def check_parameter_count(model: DecoderOnlyTransformer) -> None:
    _section("Check 4 — Parameter count")

    total = model.count_parameters()
    total_millions = total / 1_000_000

    print(f"\n  Exact parameter count : {total:,}")
    print(f"  Approximate           : {total_millions:.2f} M")

    # The expected count for this architecture is approximately 25.14 M.
    LOW  = 24_000_000
    HIGH = 26_000_000

    if total < LOW or total > HIGH:
        _fail(
            f"Parameter count {total:,} is outside the expected range "
            f"[{LOW:,}, {HIGH:,}]."
        )

    _pass(f"Parameter count is within expected range: {total:,} (~{total_millions:.2f} M).")


# ─── Check 5: Forward pass shape ─────────────────────────────────────────────

def check_forward_shape(
    model: DecoderOnlyTransformer,
    cfg: ModelConfig,
    device: torch.device,
) -> None:
    _section("Check 5 — Forward pass shape")

    model.to(device)
    model.eval()

    B, T = 2, 128   # batch=2, sequence length=128 (well within 512)

    # Create random token IDs in [0, vocab_size).
    input_ids = torch.randint(
        low=0, high=cfg.vocab_size, size=(B, T), device=device
    )   # (B, T)

    with torch.no_grad():
        logits = model(input_ids)   # (B, T, vocab_size)

    expected_shape = (B, T, cfg.vocab_size)

    print(f"\n  Input  shape : {tuple(input_ids.shape)}")
    print(f"  Output shape : {tuple(logits.shape)}")
    print(f"  Expected     : {expected_shape}")

    if logits.shape != expected_shape:
        _fail(f"Shape mismatch: got {tuple(logits.shape)}, expected {expected_shape}.")

    _pass(f"Output logits have correct shape {expected_shape}.")


# ─── Check 6: Weight tying ───────────────────────────────────────────────────

def check_weight_tying(model: DecoderOnlyTransformer) -> None:
    _section("Check 6 — Weight tying (embedding ↔ output projection)")

    emb_weight  = model.token_embedding.weight
    proj_weight = model.output_proj.weight

    print(f"\n  token_embedding.weight id : {id(emb_weight)}")
    print(f"  output_proj.weight id     : {id(proj_weight)}")
    print(f"  Same object (is)          : {emb_weight is proj_weight}")

    # Verify identity — same Python object, not just equal values.
    if emb_weight is not proj_weight:
        _fail(
            "Weight tying FAILED: output_proj.weight and "
            "token_embedding.weight are different tensors."
        )

    # Extra sanity: verify they share the same underlying storage.
    if emb_weight.data_ptr() != proj_weight.data_ptr():
        _fail(
            "Weight tying FAILED: tensors point to different memory."
        )

    # Verify the output projection still has its own independent bias.
    assert model.output_proj.bias is not None, \
        "output_proj should have a bias parameter."

    _pass(
        "output_proj.weight IS token_embedding.weight "
        "(same object, same storage). Bias remains independent."
    )


# ─── Check 7: Causal masking empirical test ───────────────────────────────────

def check_causal_masking(
    model: DecoderOnlyTransformer,
    cfg: ModelConfig,
    device: torch.device,
) -> None:
    """
    Empirical causal masking verification.

    Create two sequences that differ only at position ``change_pos``.
    Verify that logits at all positions BEFORE ``change_pos`` are identical.

    Sequence (length 8):
        Original : [BOS, 10, 20, 30, 40, 50, 60, 70]
        Modified : [BOS, 10, 20, 30, 40, 50, 60, 99]   ← last token changed

    Positions 0–6 must produce identical logits in both runs.
    """
    _section("Check 7 — Causal masking empirical test")

    model.eval()
    model.to(device)

    # ── Construct two sequences differing only at the last position ───────────
    # BOS token id = 2 (from tokenizer config).
    BOS = 2
    T   = 8
    change_pos = T - 1   # index 7 — the very last token

    seq_original = torch.tensor(
        [[BOS, 10, 20, 30, 40, 50, 60, 70]], dtype=torch.long, device=device
    )   # (1, 8)

    seq_modified = torch.tensor(
        [[BOS, 10, 20, 30, 40, 50, 60, 99]], dtype=torch.long, device=device
    )   # (1, 8) — differs only at position 7

    print(f"\n  Sequence length  : {T}")
    print(f"  Changed position : {change_pos}  (0-indexed)")
    print(f"  Positions checked for invariance : 0 .. {change_pos - 1}")

    with torch.no_grad():
        logits_orig = model(seq_original)   # (1, 8, vocab_size)
        logits_mod  = model(seq_modified)   # (1, 8, vocab_size)

    # Logits at positions 0 .. change_pos-1 must be identical.
    # (Positions at or after change_pos may differ — that is expected.)
    logits_orig_past = logits_orig[:, :change_pos, :]   # (1, 7, vocab_size)
    logits_mod_past  = logits_mod[:, :change_pos, :]    # (1, 7, vocab_size)

    max_diff = (logits_orig_past - logits_mod_past).abs().max().item()
    print(f"\n  Max absolute difference at positions 0..{change_pos - 1}: {max_diff:.2e}")

    try:
        torch.testing.assert_close(
            logits_orig_past,
            logits_mod_past,
            rtol=0,
            atol=1e-5,   # tight tolerance; should be 0.0 exactly in practice
        )
    except AssertionError as err:
        _fail(
            f"Causal masking FAILED: logits at past positions changed when a "
            f"future token was modified.\n  Error: {err}"
        )

    # Sanity: positions at or after change_pos SHOULD differ (otherwise the
    # model is ignoring the changed token entirely, which is also wrong).
    logits_orig_future = logits_orig[:, change_pos:, :]
    logits_mod_future  = logits_mod[:, change_pos:, :]
    max_diff_future = (logits_orig_future - logits_mod_future).abs().max().item()
    print(f"  Max absolute difference at positions {change_pos}+    : {max_diff_future:.4f}")

    if max_diff_future < 1e-8:
        print(
            "  [WARNING] Logits at future position also did not change. "
            "This is unexpected — the model may not be using the token at all."
        )

    _pass(
        f"Logits at positions 0..{change_pos - 1} are UNCHANGED after modifying "
        f"position {change_pos}. Causal masking is working correctly."
    )


# ─── Check 8: Context length overflow ────────────────────────────────────────

def check_context_overflow(
    model: DecoderOnlyTransformer,
    cfg: ModelConfig,
    device: torch.device,
) -> None:
    _section("Check 8 — Context length overflow guard")

    T_overflow = cfg.context_length + 1   # 513 — one beyond the limit

    input_ids = torch.randint(
        low=0, high=cfg.vocab_size, size=(1, T_overflow), device=device
    )

    print(f"\n  Attempting forward pass with T={T_overflow} (limit={cfg.context_length}) ...")

    raised_value_error = False
    error_message = ""

    try:
        with torch.no_grad():
            _ = model(input_ids)
    except ValueError as exc:
        raised_value_error = True
        error_message = str(exc)

    if not raised_value_error:
        _fail(
            f"Expected ValueError for T={T_overflow} > context_length="
            f"{cfg.context_length}, but no error was raised."
        )

    print(f"  ValueError raised as expected.")
    print(f"  Error message: {error_message}")

    _pass(
        f"ValueError is correctly raised when T={T_overflow} exceeds "
        f"context_length={cfg.context_length}."
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run all validation checks and print a final summary."""
    print("=" * 60)
    print("  Phase 2 — Decoder-Only Transformer Architecture Validation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    print(f"  PyTorch: {torch.__version__}")

    results: dict[str, bool] = {}

    def run(name: str, fn, *args):
        try:
            result = fn(*args)
            results[name] = True
            return result
        except Exception as exc:
            results[name] = False
            print(f"\n  [FAIL] {name}: {exc}")
            return None

    # ── Run all checks ────────────────────────────────────────────────────────
    cfg   = run("Config loading",   check_config_loading)
    if cfg is None:
        print("\n  Cannot continue without a valid config. Aborting.")
        sys.exit(1)

    model = run("Model instantiation", check_instantiation, cfg)
    if model is None:
        print("\n  Cannot continue without a model. Aborting.")
        sys.exit(1)

    run("Architecture print",       check_architecture_print,  model)
    run("Parameter count",          check_parameter_count,     model)
    run("Forward pass shape",       check_forward_shape,       model, cfg, device)
    run("Weight tying",             check_weight_tying,        model)
    run("Causal masking",           check_causal_masking,      model, cfg, device)
    run("Context overflow",         check_context_overflow,    model, cfg, device)

    # ── Summary ───────────────────────────────────────────────────────────────
    _section("Summary")
    all_passed = True
    check_names = [
        "Config loading",
        "Model instantiation",
        "Architecture print",
        "Parameter count",
        "Forward pass shape",
        "Weight tying",
        "Causal masking",
        "Context overflow",
    ]
    for name in check_names:
        status = results.get(name, False)
        icon   = "[PASS]" if status else "[FAIL]"
        print(f"  {icon} {name}")
        if not status:
            all_passed = False

    print()
    if all_passed:
        print("  All checks PASSED.  Phase 2 architecture is complete.")
    else:
        print("  Some checks FAILED.  Review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
