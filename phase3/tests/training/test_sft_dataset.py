"""
test_sft_dataset.py
===================
Automated tests for SFTDataset, SFTCollator, and masked_cross_entropy.

Verifies:
  1. Sequence format: [BOS] + input_tok + [EOS] + target_tok + [EOS]
  2. input_ids = raw_seq[:-1], labels = raw_seq[1:]
  3. Loss mask: 0 for BOS/input/EOS_sep, 1 for target/EOS_final
  4. Padding mask is 0 when collating variable-length records
  5. masked_cross_entropy ignores mask=0 positions
  6. SFTDataset loads the correct number of records per split
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import tempfile

import pytest
import torch
from tokenizers import Tokenizer

from phase3.scripts.training.sft_dataset import (
    SFTDataset,
    SFTCollator,
    masked_cross_entropy,
    _BOS_ID,
    _EOS_ID,
    _PAD_ID,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

TOKENIZER_PATH_HI = "hindi/tokenizer/models/bpe_v10000.json"
TOKENIZER_PATH_NE = "nepali/tokenizer/models/bpe_v10000.json"
JSONL_HI = "phase3/outputs/reasoning_hi.jsonl"
JSONL_NE = "phase3/outputs/reasoning_ne.jsonl"


def make_temp_jsonl(records):
    """Write a list of dicts to a temp JSONL file and return its path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", encoding="utf-8", delete=False
    )
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    f.flush()
    return f.name


# ── Helper: build one SFTDataset item directly ────────────────────────────────

def make_single_item(input_text: str, target_text: str, split: str = "train"):
    """Load a tiny single-record dataset and return the first item."""
    tmp = make_temp_jsonl([{
        "input_text": input_text,
        "target_text": target_text,
        "split": split,
        "task_family": "test",
    }])
    ds = SFTDataset(tmp, TOKENIZER_PATH_HI, split=split, max_seq_len=512)
    os.unlink(tmp)
    assert len(ds) == 1, "Expected exactly 1 record"
    return ds[0]


# ── Tests: sequence construction ─────────────────────────────────────────────

class TestSequenceFormat:

    def test_bos_at_input_start(self):
        item = make_single_item("आरव की ऊँचाई अधिक है।", "आरव")
        assert item["input_ids"][0].item() == _BOS_ID, "input_ids[0] must be [BOS]"

    def test_eos_sep_in_labels(self):
        item = make_single_item("आरव की ऊँचाई अधिक है।", "आरव")
        K = item["num_input_tokens"]
        assert item["labels"][K].item() == _EOS_ID, (
            f"labels[K={K}] must be [EOS] separator"
        )

    def test_eos_final_at_end_of_labels(self):
        item = make_single_item("आरव की ऊँचाई अधिक है।", "आरव")
        K = item["num_input_tokens"]
        M = item["num_target_tokens"]
        final_pos = K + 1 + M
        assert item["labels"][final_pos].item() == _EOS_ID, (
            f"labels[{final_pos}] must be final [EOS]"
        )

    def test_input_ids_and_labels_same_length(self):
        item = make_single_item("आरव", "स्वाति")
        assert item["input_ids"].shape == item["labels"].shape

    def test_labels_is_input_ids_shifted_by_one(self):
        """labels = raw_seq[1:] while input_ids = raw_seq[:-1]."""
        item = make_single_item("आरव", "स्वाति")
        # input_ids[1:] should equal labels[:-1]
        assert torch.equal(item["input_ids"][1:], item["labels"][:-1]), (
            "labels must be raw_seq[1:], i.e., input_ids shifted left by 1"
        )


# ── Tests: loss mask values ───────────────────────────────────────────────────

class TestLossMask:

    def test_bos_region_mask_is_zero(self):
        item = make_single_item("आरव की ऊँचाई स्वाति से कम है।", "स्वाति की ऊँचाई अधिक है।")
        K = item["num_input_tokens"]
        # labels indices 0..K (inclusive) must all be 0
        for i in range(K + 1):
            assert item["loss_mask"][i].item() == 0, (
                f"loss_mask[{i}] should be 0 (input/BOS/EOS_sep region)"
            )

    def test_target_region_mask_is_one(self):
        item = make_single_item("आरव की ऊँचाई स्वाति से कम है।", "स्वाति की ऊँचाई अधिक है।")
        K = item["num_input_tokens"]
        M = item["num_target_tokens"]
        for i in range(K + 1, K + 1 + M + 1):
            assert item["loss_mask"][i].item() == 1, (
                f"loss_mask[{i}] should be 1 (target/EOS_final region)"
            )

    def test_active_count_equals_M_plus_1(self):
        item = make_single_item("आरव", "स्वाति")
        M = item["num_target_tokens"]
        active = item["loss_mask"].sum().item()
        assert active == M + 1, f"Active positions should be {M+1}, got {active}"

    def test_total_mask_length_equals_seq_len(self):
        item = make_single_item("आरव", "स्वाति")
        assert item["loss_mask"].shape == item["input_ids"].shape

    def test_minimal_example(self):
        """
        Minimal synthetic test with a very short sequence to verify positions
        by hand without relying on real tokenisation ambiguity.
        """
        item = make_single_item("?", "?")
        # Tokeniser may split "?" as a single token or multiple; just check structure.
        K = item["num_input_tokens"]
        M = item["num_target_tokens"]
        mask = item["loss_mask"].tolist()
        # Input region (0..K) all zero
        assert all(m == 0 for m in mask[:K + 1])
        # Target region (K+1 .. end) all one
        assert all(m == 1 for m in mask[K + 1:])


# ── Tests: collator ───────────────────────────────────────────────────────────

class TestSFTCollator:

    def test_padding_positions_have_mask_zero(self):
        """Shorter sequences get padded; padding loss_mask must be 0."""
        item_short = make_single_item("?", "?")
        item_long  = make_single_item("आरव की ऊँचाई स्वाति से कम है। किसकी ऊँचाई अधिक है?",
                                      "स्वाति की ऊँचाई अधिक है।")
        collator = SFTCollator(pad_id=_PAD_ID)
        batch = collator([item_short, item_long])

        T_max = batch["input_ids"].shape[1]
        for b in range(2):
            for t in range(T_max):
                if batch["labels"][b, t].item() == _PAD_ID:
                    assert batch["loss_mask"][b, t].item() == 0, (
                        f"Padding position [{b},{t}] must have loss_mask=0"
                    )

    def test_collator_stacks_correctly(self):
        item = make_single_item("आरव", "स्वाति")
        collator = SFTCollator(pad_id=_PAD_ID)
        batch = collator([item, item])
        assert batch["input_ids"].shape[0] == 2
        assert batch["labels"].shape[0] == 2
        assert batch["loss_mask"].shape[0] == 2

    def test_collator_output_keys(self):
        item = make_single_item("आरव", "स्वाति")
        collator = SFTCollator(pad_id=_PAD_ID)
        batch = collator([item])
        assert "input_ids" in batch
        assert "labels" in batch
        assert "loss_mask" in batch


# ── Tests: masked_cross_entropy ───────────────────────────────────────────────

class TestMaskedCrossEntropy:

    def test_raises_if_no_active_positions(self):
        logits    = torch.randn(1, 5, 100)
        labels    = torch.zeros(1, 5, dtype=torch.long)
        loss_mask = torch.zeros(1, 5, dtype=torch.long)
        with pytest.raises(ValueError, match="all loss_mask entries are 0"):
            masked_cross_entropy(logits, labels, loss_mask)

    def test_masked_positions_do_not_affect_loss(self):
        """Changing logits at mask=0 positions must not change the loss."""
        torch.manual_seed(0)
        B, T, V = 1, 8, 50
        logits  = torch.randn(B, T, V)
        labels  = torch.randint(0, V, (B, T))
        mask    = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.long)

        loss1 = masked_cross_entropy(logits, labels, mask)

        # Scramble the masked-out logits
        logits_modified = logits.clone()
        logits_modified[0, :3, :] = torch.randn(3, V)
        loss2 = masked_cross_entropy(logits_modified, labels, mask)

        assert torch.isclose(loss1, loss2, atol=1e-5), (
            f"Loss changed when modifying masked-out positions: {loss1} vs {loss2}"
        )

    def test_loss_is_scalar(self):
        logits    = torch.randn(2, 10, 100)
        labels    = torch.randint(0, 100, (2, 10))
        loss_mask = torch.ones(2, 10, dtype=torch.long)
        loss = masked_cross_entropy(logits, labels, loss_mask)
        assert loss.shape == torch.Size([]), "Loss must be a scalar"

    def test_full_mask_matches_f_cross_entropy(self):
        """When all positions are active, result should match standard CE."""
        torch.manual_seed(42)
        B, T, V = 2, 6, 50
        logits    = torch.randn(B, T, V)
        labels    = torch.randint(2, V, (B, T))   # avoid PAD_ID=1
        loss_mask = torch.ones(B, T, dtype=torch.long)

        import torch.nn.functional as F
        expected = F.cross_entropy(
            logits.view(B * T, V), labels.view(B * T),
            ignore_index=_PAD_ID, reduction="mean"
        )
        actual = masked_cross_entropy(logits, labels, loss_mask)
        assert torch.isclose(actual, expected, atol=1e-5), (
            f"Full-mask CE mismatch: expected {expected.item():.6f}, got {actual.item():.6f}"
        )


# ── Tests: dataset loads correct split counts ─────────────────────────────────

class TestDatasetSplitCounts:

    def test_hindi_train_count(self):
        ds = SFTDataset(JSONL_HI, TOKENIZER_PATH_HI, split="train")
        assert len(ds) == 5600, f"Expected 5600 train records, got {len(ds)}"

    def test_hindi_validation_count(self):
        ds = SFTDataset(JSONL_HI, TOKENIZER_PATH_HI, split="val")
        assert len(ds) == 700, f"Expected 700 val records, got {len(ds)}"

    def test_hindi_test_count(self):
        ds = SFTDataset(JSONL_HI, TOKENIZER_PATH_HI, split="test")
        assert len(ds) == 700, f"Expected 700 test records, got {len(ds)}"

    def test_nepali_train_count(self):
        ds = SFTDataset(JSONL_NE, TOKENIZER_PATH_NE, split="train")
        assert len(ds) == 5600, f"Expected 5600 train records, got {len(ds)}"

    def test_no_records_dropped_for_length(self):
        """All 7000-record reasoning files should fit within 512 tokens."""
        ds_hi = SFTDataset(JSONL_HI, TOKENIZER_PATH_HI, split=None, max_seq_len=512)
        assert ds_hi._dropped == 0, f"Hindi: {ds_hi._dropped} records dropped for length"
        ds_ne = SFTDataset(JSONL_NE, TOKENIZER_PATH_NE, split=None, max_seq_len=512)
        assert ds_ne._dropped == 0, f"Nepali: {ds_ne._dropped} records dropped for length"
