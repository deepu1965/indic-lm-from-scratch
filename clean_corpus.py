#!/usr/bin/env python3
"""
clean_corpus.py — Reusable JSONL corpus cleaning pipeline (LMA project).

Works for any language (Hindi, Nepali, ...) purely by changing config.yaml.
No language-specific paths or logic are hard-coded in this file.

Stages
------
Public data (public/pdf, public/speech, public/web):
    1. unicode_normalization
    2. non_language_character_removal
    3. latin_ratio_filtering

Manual data (manual/):
    1. unicode_normalization
    2. mediawiki_markup_cleanup      ← removes leftover image/file fragments
    3. non_language_character_removal
    4. length_filtering
    5. line_structure_filtering
    6. repetition_filtering
    7. latin_ratio_filtering

Run:
    python clean_corpus.py --config config_hindi.yaml
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

import yaml


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

class Config:
    """Thin wrapper around the loaded config.yaml with convenience accessors."""

    def __init__(self, raw: Dict[str, Any]):
        self.raw = raw

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load a YAML config file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def get(self, *keys, default=None):
        """Fetch a nested value, e.g. cfg.get('thresholds', 'length_filtering', 'min_chars')."""
        node = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


# --------------------------------------------------------------------------
# Stage 1: Unicode normalization
# --------------------------------------------------------------------------

class UnicodeNormalizer:
    """Applies consistent Unicode normalization and whitespace cleanup.

    Whitespace handling collapses runs of horizontal whitespace and trims
    trailing spaces around line breaks, but preserves line/paragraph
    structure (single vs. blank-line breaks) since manual-data filters
    downstream (line structure, repetition) depend on it.
    """

    _HORIZONTAL_WS_RE = re.compile(r"[ \t\u00A0\u200b]+")
    _WS_AROUND_NEWLINE_RE = re.compile(r"[ \t]*\n[ \t]*")
    _EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")

    def __init__(self, form: str = "NFC"):
        self.form = form

    def normalize(self, text: str) -> str:
        """Return Unicode-normalized text with tidy, structure-preserving whitespace."""
        text = unicodedata.normalize(self.form, text)
        text = self._HORIZONTAL_WS_RE.sub(" ", text)
        text = self._WS_AROUND_NEWLINE_RE.sub("\n", text)
        text = self._EXCESS_BLANK_LINES_RE.sub("\n\n", text)
        return text.strip()


# --------------------------------------------------------------------------
# Stage 2 (manual only): MediaWiki image/file markup fragment cleanup
# --------------------------------------------------------------------------

class MediaWikiMarkupCleaner:
    """Removes leftover MediaWiki image/file markup fragments from plain text.

    After mwparserfromhell strips wikitext, [[File:...]] / [[Image:...]] links
    sometimes leave behind their pipe-separated option tokens as raw text, e.g.:
        "300 px|thumb|राफेल द्वारा पारनासस"
        "thumb|left|राफेल द्वारा पारनासस"
        "250x180px|thumb|राफेल द्वारा पारनासस"
        "upright=1.5|thumb|right|Caption"
    This cleaner removes only those specific token patterns, preserving the
    meaningful text/caption that follows them.  Normal Hindi/Nepali prose
    cannot match these patterns because they require a bare pipe character ('|')
    immediately after an image keyword or a numeric size token — neither of
    which appears in natural language.

    Applied ONLY to manual (Wikipedia/Wikisource) data, not to public/Sangraha
    data, since only MediaWiki dumps produce these artefacts.
    """

    # Pattern 1 — optional leading size token followed by one or more pipe-
    # separated MediaWiki image option keywords.
    # Matches: "thumb|", "300 px|thumb|left|", "upright=1.5|thumb|right|"
    _IMAGE_OPTION_KEYWORDS = (
        r"thumb|thumbnail|frame|frameless|border|baseline|sub|super|"
        r"top|text-top|middle|bottom|text-bottom|upright|left|right|center|none"
    )
    _MEDIA_FRAGMENT_RE = re.compile(
        r"(?:"
        r"(?:\d+\s*(?:x\d+)?\s*px|upright\s*(?:=\s*[\d.]+)?)\s*\|\s*"  # size token
        r")*"
        r"(?:(?:" + _IMAGE_OPTION_KEYWORDS + r")\s*\|\s*)+",  # one or more keyword|
        re.IGNORECASE,
    )

    # Pattern 2 — a lone bare "N px|" or "NxNpx|" size token with no keyword
    # (catches "300 px|" that can appear without an explicit keyword).
    _SIZE_ONLY_FRAGMENT_RE = re.compile(
        r"\d+\s*(?:x\d+)?\s*px\s*\|",
        re.IGNORECASE,
    )

    def clean(self, text: str) -> str:
        """Remove MediaWiki image/file option tokens, then tidy whitespace."""
        text = self._MEDIA_FRAGMENT_RE.sub("", text)
        text = self._SIZE_ONLY_FRAGMENT_RE.sub("", text)
        # Collapse runs of spaces/tabs and remove any leading space per line
        # that result from the deletion above.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"^ ", "", text, flags=re.MULTILINE)
        return text


# --------------------------------------------------------------------------
# Stage 3: Non-language / garbage character removal
# --------------------------------------------------------------------------

class CharacterFilter:
    """Conservative, script-aware garbage-character removal.

    Two independent decisions per character:
      1. Control/format/unassigned/private-use/surrogate characters (Unicode
         category "C") are always dropped, except ZWJ/ZWNJ which are kept
         when Devanagari is an allowed script (needed for correct conjunct
         rendering).
      2. Letters (L) and combining marks (M) are kept only if their code
         point falls inside one of the `allowed_scripts` ranges (from YAML).
         Numbers (N), punctuation (P), and symbols (S) are script-agnostic
         and are kept as whole categories, gated only by the
         preserve_digits / preserve_punctuation / preserve_symbols flags —
         this is what keeps digits, currency symbols, math operators, %,
         @, #, quotes, brackets, hyphens/dashes, slashes, and URL/email
         punctuation intact regardless of script.

    This is deliberately NOT "keep only Devanagari": Latin letters (and any
    other configured script) are preserved so technical terms, names, and
    abbreviations survive. Only letters/marks from scripts that are not in
    `allowed_scripts` are dropped as unrelated-script garbage.
    """

    # Zero-width joiner / non-joiner: essential for correct Devanagari rendering.
    _ESSENTIAL_FORMAT_CHARS = {"\u200c", "\u200d"}
    _ALWAYS_KEEP = {"\n", "\t", " "}

    # Unicode code-point ranges (inclusive) per script. Kept minimal and
    # standard-library only (no external script-detection dependency needed
    # for the two scripts currently in use).
    _SCRIPT_RANGES: Dict[str, Tuple[Tuple[int, int], ...]] = {
        "devanagari": (
            (0x0900, 0x097F),  # Devanagari (incl. vowel signs, digits, danda)
            (0xA8E0, 0xA8FF),  # Devanagari Extended
            (0x1CD0, 0x1CFF),  # Vedic Extensions (combining marks)
        ),
        "latin": (
            (0x0041, 0x005A),  # Basic Latin A-Z
            (0x0061, 0x007A),  # Basic Latin a-z
            (0x00C0, 0x00FF),  # Latin-1 Supplement letters
            (0x0100, 0x024F),  # Latin Extended-A/B
        ),
    }

    def __init__(
        self,
        allowed_scripts=None,
        preserve_digits: bool = True,
        preserve_punctuation: bool = True,
        preserve_symbols: bool = True,
        extra_keep_chars: str = "",
        extra_remove_chars: str = "",
    ):
        self.allowed_scripts = [str(s).lower() for s in (allowed_scripts or ["devanagari", "latin"])]
        self.preserve_digits = preserve_digits
        self.preserve_punctuation = preserve_punctuation
        self.preserve_symbols = preserve_symbols
        self.extra_keep = set(extra_keep_chars or "")
        self.extra_remove = set(extra_remove_chars or "")
        self._ranges: Tuple[Tuple[int, int], ...] = tuple(
            r for script in self.allowed_scripts for r in self._SCRIPT_RANGES.get(script, ())
        )

    def _script_allowed(self, codepoint: int) -> bool:
        return any(lo <= codepoint <= hi for lo, hi in self._ranges)

    def clean(self, text: str) -> str:
        """Strip non-language garbage characters from text, preserving meaning."""
        out = []
        for ch in text:
            if ch in self.extra_remove:
                continue
            if ch in self.extra_keep or ch in self._ALWAYS_KEEP:
                out.append(ch)
                continue
            if ch in self._ESSENTIAL_FORMAT_CHARS:
                if "devanagari" in self.allowed_scripts:
                    out.append(ch)
                continue

            cat = unicodedata.category(ch)

            if cat[0] == "C":  # Cc, Cf, Co, Cs, Cn -> drop (control/format/unassigned/etc.)
                continue
            if cat == "Zl" or cat == "Zp":  # line/paragraph separators -> normalize to newline
                out.append("\n")
                continue
            if cat == "Zs":  # other Unicode space separators -> normalize to plain space
                out.append(" ")
                continue

            if cat[0] == "L" or cat[0] == "M":  # letters, combining marks -> script-gated
                if self._script_allowed(ord(ch)):
                    out.append(ch)
                continue

            if cat[0] == "N":  # digits/numbers (ASCII and Indic) -> language-agnostic
                if self.preserve_digits:
                    out.append(ch)
                continue

            if cat[0] == "P":  # punctuation, quotes, brackets, slashes, hyphens/dashes, @, #
                if self.preserve_punctuation:
                    out.append(ch)
                continue

            if cat[0] == "S":  # symbols: currency, math operators, %, other useful symbols
                if self.preserve_symbols:
                    out.append(ch)
                continue

            # Anything else falls through and is dropped conservatively.
        return "".join(out)


# --------------------------------------------------------------------------
# Stage 4 (manual only): Length filtering
# --------------------------------------------------------------------------

class LengthFilter:
    """Rejects documents shorter than configured minimum chars/words."""

    def __init__(self, min_chars: int = 0, min_words: int = 0):
        self.min_chars = min_chars
        self.min_words = min_words

    def passes(self, text: str) -> bool:
        """Return True if the document meets both minimum thresholds."""
        if len(text) < self.min_chars:
            return False
        if len(text.split()) < self.min_words:
            return False
        return True


# --------------------------------------------------------------------------
# Stage 5 (manual only): Line structure filtering
# --------------------------------------------------------------------------

class LineStructureFilter:
    """Rejects documents dominated by extremely short lines.

    Single O(N) pass over lines: tracks the short-line ratio and the
    longest run of consecutive short lines.
    """

    def __init__(
        self,
        short_line_max_chars: int = 3,
        max_short_line_ratio: float = 0.5,
        max_consecutive_short_lines: int = 5,
    ):
        self.short_line_max_chars = short_line_max_chars
        self.max_short_line_ratio = max_short_line_ratio
        self.max_consecutive_short_lines = max_consecutive_short_lines

    def passes(self, text: str) -> bool:
        """Return True if the document's line structure looks healthy."""
        lines = [l for l in text.split("\n") if l.strip() != ""]
        total = len(lines)
        if total == 0:
            return False

        short_count = 0
        consecutive = 0
        max_consecutive = 0
        for line in lines:
            if len(line.strip()) <= self.short_line_max_chars:
                short_count += 1
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0

        if (short_count / total) > self.max_short_line_ratio:
            return False
        if max_consecutive > self.max_consecutive_short_lines:
            return False
        return True


# --------------------------------------------------------------------------
# Stage 6 (manual only): Repetition filtering
# --------------------------------------------------------------------------

class RepetitionFilter:
    """Rejects obviously repetitive/garbage documents using repeated-word,
    repeated-trigram, and (optionally) repeated-paragraph ratios.

    Uses Counters/sets only (no pairwise document comparison), so each
    document is processed in roughly O(N) time/space.
    """

    def __init__(
        self,
        max_repeated_word_ratio: float = 1.0,
        max_repeated_trigram_ratio: float = 1.0,
        max_repeated_paragraph_ratio: Optional[float] = None,
    ):
        self.max_repeated_word_ratio = max_repeated_word_ratio
        self.max_repeated_trigram_ratio = max_repeated_trigram_ratio
        self.max_repeated_paragraph_ratio = max_repeated_paragraph_ratio

    @staticmethod
    def _repeated_ratio(counter: Counter, total: int) -> float:
        return 1.0 - (len(counter) / total) if total else 0.0

    def passes(self, text: str) -> bool:
        """Return True if the document is not dominated by repeated content."""
        words = text.split()
        if words:
            word_ratio = self._repeated_ratio(Counter(words), len(words))
            if word_ratio > self.max_repeated_word_ratio:
                return False

        if len(words) >= 3:
            trigram_counts = Counter(zip(words, words[1:], words[2:]))
            total_trigrams = max(len(words) - 2, 0)
            tri_ratio = self._repeated_ratio(trigram_counts, total_trigrams)
            if tri_ratio > self.max_repeated_trigram_ratio:
                return False

        if self.max_repeated_paragraph_ratio is not None:
            paragraphs = [p for p in text.split("\n\n") if p.strip()]
            if len(paragraphs) >= 2:
                para_ratio = self._repeated_ratio(Counter(paragraphs), len(paragraphs))
                if para_ratio > self.max_repeated_paragraph_ratio:
                    return False

        return True


# --------------------------------------------------------------------------
# Stage 7: Latin-heavy sentence filter
# --------------------------------------------------------------------------

class LatinRatioFilter:
    """Drops sentences that are predominantly Latin/English from a document.

    Operates at the SENTENCE level, not the document level, so mixed
    Hindi/Nepali + English text is preserved: only the sentences that
    are themselves predominantly Latin are removed.  Sentences with
    no letters at all (e.g. a line of digits or punctuation) are kept
    unchanged since there is nothing meaningful to filter.

    What counts as "Latin":
        Any character in Unicode categories L* whose code point falls in
        the Basic Latin letter range (A-Z / a-z, U+0041..U+007A) or the
        wider Latin Extended ranges (U+00C0..U+024F).  This does NOT
        include digits, punctuation, or Devanagari letters.

    The ratio is computed as:
        latin_letter_count / total_letter_count   (letters only, not spaces)

    If total_letter_count == 0 the sentence is kept (no letters to judge).

    Example outcomes with max_latin_ratio=0.80:
        "\u092d\u093e\u0930\u0924 \u092e\u0947\u0902 AI \u0915\u093e \u0909\u092a\u092f\u094b\u0917 \u092c\u0922\u093c \u0930\u0939\u093e \u0939\u0948"  -> kept   (AI = 2 Latin out of ~14 letters)
        "This is a completely English sentence."    -> dropped (26/26 = 100% Latin)
        "COVID-19 \u0915\u0947 \u092c\u093e\u0926 \u0938\u094d\u0935\u093e\u0938\u094d\u0925\u094d\u092f \u0938\u0947\u0935\u093e\u090f\u0902"              -> kept   (COVID = 5 Latin out of ~17)
    """

    # Latin letter code-point ranges (letters only, no digits/punctuation).
    _LATIN_RANGES = (
        (0x0041, 0x005A),   # Basic Latin A–Z
        (0x0061, 0x007A),   # Basic Latin a–z
        (0x00C0, 0x00FF),   # Latin-1 Supplement letters
        (0x0100, 0x024F),   # Latin Extended-A/B
    )

    # Simple sentence splitter: split on common sentence-ending punctuation
    # (Devanagari danda \u0964, double danda \u0965, and ASCII .!?).
    # We keep the delimiter attached to the preceding segment so the
    # reconstructed text looks natural.
    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\u0964\u0965])\s+")

    def __init__(self, max_latin_ratio: float = 0.80, enabled: bool = True):
        """Initialize the filter.

        Args:
            max_latin_ratio: Sentences where the fraction of Latin letters
                (over all letters) exceeds this threshold are dropped.
                0.80 means "drop a sentence if more than 80% of its letters
                are Latin".  Set to 1.0 to effectively disable.
            enabled: When False the filter is a no-op (all sentences pass).
        """
        self.max_latin_ratio = max_latin_ratio
        self.enabled = enabled

    def _is_latin(self, cp: int) -> bool:
        """Return True if the code point is a Latin letter."""
        return any(lo <= cp <= hi for lo, hi in self._LATIN_RANGES)

    def _latin_ratio(self, text: str) -> float:
        """Return the fraction of letters in `text` that are Latin."""
        total_letters = 0
        latin_letters = 0
        for ch in text:
            if unicodedata.category(ch)[0] == "L":  # any letter category
                total_letters += 1
                if self._is_latin(ord(ch)):
                    latin_letters += 1
        if total_letters == 0:
            return 0.0  # no letters → ratio undefined, treat as 0 (keep)
        return latin_letters / total_letters

    def filter_sentences(self, text: str) -> str:
        """Return `text` with predominantly-Latin sentences removed.

        Sentence boundaries are detected by a simple regex on common
        end-of-sentence punctuation.  Paragraph structure (blank lines)
        is preserved by splitting/joining on a per-paragraph basis.
        """
        if not self.enabled:
            return text

        # Process paragraph by paragraph so blank-line structure is kept.
        paragraphs = text.split("\n")
        out_paragraphs = []
        for para in paragraphs:
            if not para.strip():
                out_paragraphs.append(para)  # preserve blank/whitespace lines
                continue

            # Split the paragraph into sentences.
            sentences = self._SENTENCE_SPLIT_RE.split(para)
            kept = []
            for sent in sentences:
                if self._latin_ratio(sent) <= self.max_latin_ratio:
                    kept.append(sent)  # keep sentence
                # else: drop the predominantly-Latin sentence silently

            # Re-join kept sentences with a single space.
            rejoined = " ".join(kept).strip()
            if rejoined:
                out_paragraphs.append(rejoined)

        return "\n".join(out_paragraphs)

class StatsTracker:
    """Accumulates word/char totals before/after cleaning, split by source
    category ('public' -> Sangraha Verified, 'manual' -> Manual)."""

    CATEGORY_LABELS = {"public": "Sangraha Verified", "manual": "Manual"}

    def __init__(self):
        self.data = {
            "before": {"public": {"words": 0, "chars": 0}, "manual": {"words": 0, "chars": 0}},
            "after": {"public": {"words": 0, "chars": 0}, "manual": {"words": 0, "chars": 0}},
        }

    def add(self, stage: str, category: str, text: str) -> None:
        """Record word/char counts for a document at a given stage ('before'/'after')."""
        self.data[stage][category]["words"] += len(text.split())
        self.data[stage][category]["chars"] += len(text)

    def print_report(self) -> None:
        """Print the before/after statistics report split by source and overall."""
        for stage in ("before", "after"):
            label = "Before" if stage == "before" else "After"
            print(f"\n{label}:")
            total_words = 0
            total_chars = 0
            for category in ("public", "manual"):
                d = self.data[stage][category]
                print(f"    {self.CATEGORY_LABELS[category]}:")
                print(f"        words: {d['words']}")
                print(f"        chars: {d['chars']}")
                total_words += d["words"]
                total_chars += d["chars"]
            print(f"    Overall:")
            print(f"        words: {total_words}")
            print(f"        chars: {total_chars}")


# --------------------------------------------------------------------------
# Record processing / orchestration
# --------------------------------------------------------------------------

def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Stream a JSONL file one record at a time."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"    [warn] skipping malformed line in {path}: {e}", file=sys.stderr)


def make_id(cleaned_text: str, hash_length: Optional[int] = None) -> str:
    """Generate a deterministic id ('id_' + sha256 hash) from the cleaned text."""
    digest = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
    if hash_length:
        digest = digest[:hash_length]
    return f"id_{digest}"


class CorpusPipeline:
    """Discovers input JSONL files, runs the configured cleaning stages, and
    streams cleaned records to output — never loading the full corpus into
    memory."""

    def __init__(self, config: Config):
        self.cfg = config

        self.text_field = config.get("fields", "text_field", default="text")
        self.id_hash_length = config.get("output", "id_hash_length", default=None)
        self.overwrite = bool(config.get("output", "overwrite_existing", default=False))

        self.corpus_root = config.get("paths", "corpus_root")
        self.manual_dir = config.get("paths", "manual_dir", default="manual")
        self.public_dirs = config.get("paths", "public_dirs", default=[])
        self.output_dir = config.get("paths", "output_dir", default="cleaned")

        if not self.corpus_root:
            raise ValueError("config.yaml: paths.corpus_root is required")

        # Stage instances, built once from config.
        self.normalizer = UnicodeNormalizer(
            form=config.get("unicode_normalization", "form", default="NFC")
        )
        # MediaWiki markup cleanup — applied to manual data only (Stage 2 of
        # the manual pipeline).  No config knobs needed: the two regexes are
        # precise enough that there is no threshold to tune.
        self.mediawiki_cleaner = MediaWikiMarkupCleaner()
        self.char_filter = CharacterFilter(
            allowed_scripts=config.get("character_filtering", "allowed_scripts", default=["devanagari", "latin"]),
            preserve_digits=config.get("character_filtering", "preserve_digits", default=True),
            preserve_punctuation=config.get("character_filtering", "preserve_punctuation", default=True),
            preserve_symbols=config.get("character_filtering", "preserve_symbols", default=True),
            extra_keep_chars=config.get("character_filtering", "extra_keep_chars", default=""),
            extra_remove_chars=config.get("character_filtering", "extra_remove_chars", default=""),
        )
        self.length_filter = LengthFilter(
            min_chars=config.get("thresholds", "length_filtering", "min_chars", default=0),
            min_words=config.get("thresholds", "length_filtering", "min_words", default=0),
        )
        self.line_filter = LineStructureFilter(
            short_line_max_chars=config.get(
                "thresholds", "line_structure_filtering", "short_line_max_chars", default=3
            ),
            max_short_line_ratio=config.get(
                "thresholds", "line_structure_filtering", "max_short_line_ratio", default=0.5
            ),
            max_consecutive_short_lines=config.get(
                "thresholds", "line_structure_filtering", "max_consecutive_short_lines", default=5
            ),
        )
        self.repetition_filter = RepetitionFilter(
            max_repeated_word_ratio=config.get(
                "thresholds", "repetition_filtering", "max_repeated_word_ratio", default=1.0
            ),
            max_repeated_trigram_ratio=config.get(
                "thresholds", "repetition_filtering", "max_repeated_trigram_ratio", default=1.0
            ),
            max_repeated_paragraph_ratio=config.get(
                "thresholds", "repetition_filtering", "max_repeated_paragraph_ratio", default=None
            ),
        )

        # --- Fix 2: Latin-ratio sentence filter (configurable via YAML) ---
        # Reads thresholds.latin_ratio_filtering from config.  Falls back
        # gracefully to "disabled" if the key is absent, so existing configs
        # without this section continue to work unchanged.
        latin_cfg = config.get("thresholds", "latin_ratio_filtering") or {}
        self.latin_filter = LatinRatioFilter(
            max_latin_ratio=latin_cfg.get("max_latin_ratio", 0.80),
            enabled=latin_cfg.get("enabled", True),
        )

        self.stats = StatsTracker()

    # -- file discovery ----------------------------------------------------

    def _discover_files(self) -> Iterable[Tuple[str, str]]:
        """Yield (filepath, category) for every input JSONL file. category is
        'manual' or 'public'."""
        manual_glob = os.path.join(self.corpus_root, self.manual_dir, "*.jsonl")
        for path in sorted(glob.glob(manual_glob)):
            yield path, "manual"

        for sub_dir in self.public_dirs:
            public_glob = os.path.join(self.corpus_root, sub_dir, "*.jsonl")
            for path in sorted(glob.glob(public_glob)):
                yield path, "public"

    # -- output naming -------------------------------------------------------

    def _output_path(self, input_path: str, category: str) -> str:
        """Build a deterministic, collision-free output path for an input file."""
        rel = os.path.relpath(input_path, self.corpus_root)
        safe_name = rel.replace(os.sep, "__")
        return os.path.join(self.corpus_root, self.output_dir, f"cleaned__{safe_name}")

    # -- per-record cleaning --------------------------------------------------

    def _clean_text(self, text: str, category: str) -> Optional[str]:
        """Run the appropriate stage sequence for the category.

        Public pipeline:  normalize → char_filter → latin_filter
        Manual pipeline:  normalize → mediawiki_cleanup → char_filter
                          → length_filter → line_filter → repetition_filter
                          → latin_filter

        Returns the cleaned text, or None if the document was filtered out.
        """
        # Stage 1: Unicode normalization (all data)
        text = self.normalizer.normalize(text)

        if category == "manual":
            # Stage 2 (manual only): remove leftover MediaWiki image-option
            # tokens that mwparserfromhell leaves behind after stripping.
            text = self.mediawiki_cleaner.clean(text)

        # Stage 3 (public: stage 2): character filtering (all data)
        text = self.char_filter.clean(text)

        if category == "manual":
            # Stage 4: length filter
            if not self.length_filter.passes(text):
                return None
            # Stage 5: line structure filter
            if not self.line_filter.passes(text):
                return None
            # Stage 6: repetition filter
            if not self.repetition_filter.passes(text):
                return None

        # Stage 7 (public: stage 3): drop predominantly-Latin sentences
        text = self.latin_filter.filter_sentences(text)

        return text

    def _build_output_record(self, record: Dict[str, Any], cleaned_text: str, category: str) -> Dict[str, Any]:
        """Construct the output record: updated text/id/source, other metadata preserved."""
        out = dict(record)  # preserve other metadata (language, project, type, ...)
        out.pop("id", None)  # never keep the original id
        out[self.text_field] = cleaned_text
        out["id"] = make_id(cleaned_text, self.id_hash_length)
        if category == "public":
            out["source"] = "sangraha verified"
        else:
            out["source"] = record.get("source", "manual")
        return out

    # -- stats for previously-cleaned, skipped files ---------------------------

    def _add_skipped_stats(self, input_path: str, output_path: str, category: str) -> None:
        """A cleaned output file already exists and is being left untouched.
        Stream both files record-by-record (never loading either fully into
        memory) so the printed statistics still include this file, without
        re-cleaning or overwriting anything."""
        if not os.path.exists(input_path):
            print(
                f"    [warn] cannot find input file {input_path} for existing output "
                f"{output_path}; 'before cleaning' stats for this file are omitted",
                file=sys.stderr,
            )
        else:
            for record in iter_jsonl(input_path):
                raw_text = record.get(self.text_field, "")
                if isinstance(raw_text, str) and raw_text:
                    self.stats.add("before", category, raw_text)

        for record in iter_jsonl(output_path):
            cleaned_text = record.get(self.text_field, "")
            if isinstance(cleaned_text, str) and cleaned_text:
                self.stats.add("after", category, cleaned_text)

    # -- main run --------------------------------------------------------------

    def run(self) -> None:
        """Execute the full pipeline: discover files, clean, write output, report stats."""
        out_dir = os.path.join(self.corpus_root, self.output_dir)
        os.makedirs(out_dir, exist_ok=True)

        files = list(self._discover_files())
        if not files:
            print(f"No input JSONL files found under {self.corpus_root}")
            return

        for input_path, category in files:
            output_path = self._output_path(input_path, category)

            if os.path.exists(output_path) and not self.overwrite:
                print(f"[skip] {output_path} already exists (set output.overwrite_existing: true to replace)")
                self._add_skipped_stats(input_path, output_path, category)
                continue

            print(f"[processing] {input_path}  ({category})")
            kept, dropped = 0, 0

            with open(output_path, "w", encoding="utf-8") as out_f:
                for record in iter_jsonl(input_path):
                    raw_text = record.get(self.text_field, "")
                    if not isinstance(raw_text, str) or not raw_text:
                        dropped += 1
                        continue

                    self.stats.add("before", category, raw_text)

                    cleaned_text = self._clean_text(raw_text, category)
                    if cleaned_text is None or not cleaned_text.strip():
                        dropped += 1
                        continue

                    self.stats.add("after", category, cleaned_text)

                    out_record = self._build_output_record(record, cleaned_text, category)
                    out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                    kept += 1

            print(f"    kept={kept} dropped={dropped} -> {output_path}")

        self.stats.print_report()


# --------------------------------------------------------------------------
# CLI entrypoint
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LMA multilingual corpus cleaning pipeline")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    config = Config.load(args.config)
    pipeline = CorpusPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()