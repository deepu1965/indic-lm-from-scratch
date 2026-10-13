"""
tokenizer_utils.py
==================
Shared whitespace preprocessing and decoding utilities for Hindi and Nepali BPE tokenizers.

Design
------
The HuggingFace ``Whitespace()`` pre-tokenizer is memory-efficient but discards
whitespace boundary information.  To preserve spaces, newlines, and tabs for
reversible decoding we use three protected special tokens:

    <SPACE>   — represents one original U+0020 space
    <NEWLINE> — represents one original U+000A newline
    <TAB>     — represents one original U+0009 tab

``preprocess_text()`` replaces each whitespace character with a delimiter-
surrounded marker in a single regex pass (no sequential replacements that could
interfere with each other).  The surrounding delimiter spaces are real ASCII
spaces that ``Whitespace()`` uses as split boundaries; they are *not* data.

``custom_decode()`` inverts the process: concatenates all non-structural BPE
subword tokens and then substitutes markers back to their original characters.

Limitations
-----------
* If the raw text literally contains the strings ``<SPACE>``, ``<NEWLINE>``, or
  ``<TAB>`` (e.g. in HTML/XML or code), those strings will be incorrectly
  expanded to whitespace during decoding.  This is negligible for Hindi/Nepali
  prose corpora.
* If a token is encoded as ``[UNK]`` (out-of-vocabulary character), it cannot
  be recovered.  ``custom_decode()`` substitutes the literal string ``[UNK]``
  in that position, so the round-trip is *not* exact when unknown tokens exist.
  The caller is responsible for checking ``unk_count`` before asserting exact
  reconstruction.
"""

import re

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_WS_PATTERN = re.compile(r'[ \n\t]')

_WS_MAP: dict[str, str] = {
    " ":  " <SPACE> ",
    "\n": " <NEWLINE> ",
    "\t": " <TAB> ",
}


def _ws_replacer(match: re.Match) -> str:
    return _WS_MAP[match.group(0)]


def preprocess_text(text: str) -> str:
    """Normalize line endings and encode whitespace explicitly in a single pass.

    Each whitespace character is replaced by a delimiter-surrounded marker:
        ' '  ->  ' <SPACE> '
        '\\n' ->  ' <NEWLINE> '
        '\\t' ->  ' <TAB> '

    The surrounding real spaces act as split delimiters for the Whitespace
    pre-tokenizer and are NOT themselves data characters.  The regex processes
    every whitespace character exactly once with no risk of cascading
    replacements.

    Parameters
    ----------
    text : str
        Raw document text from the corpus.

    Returns
    -------
    str
        Preprocessed text suitable for ``Tokenizer.encode()``.
    """
    # Normalise Windows/old-Mac line endings first (not part of the data pass)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Single-pass replacement: each [ \\n\\t] character is handled exactly once.
    return _WS_PATTERN.sub(_ws_replacer, text)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

# Tokens produced by the HuggingFace BPE model that represent structural
# information and should NOT be included in the reconstructed output.
_SKIP_TOKENS: frozenset[str] = frozenset({"[PAD]", "[BOS]", "[EOS]"})


def custom_decode(tokenizer, token_ids: list) -> str:
    """Reconstruct the original (normalised) text from a list of token IDs.

    Algorithm
    ---------
    1. Convert IDs to piece strings via ``id_to_token``.
    2. Skip structural tokens ([PAD], [BOS], [EOS]).
    3. Keep [UNK] literally — it signals an irrecoverable character.
    4. Concatenate all remaining pieces (BPE subwords join naturally).
    5. Replace whitespace markers with real whitespace characters.

    Parameters
    ----------
    tokenizer : tokenizers.Tokenizer
        The trained BPE tokenizer (provides ``id_to_token``).
    token_ids : list[int]
        Token IDs from ``Tokenizer.encode(...).ids``.

    Returns
    -------
    str
        Reconstructed text.  Equal to the *normalised* original (after CRLF
        normalisation) when no ``[UNK]`` tokens are present.
    """
    pieces = []
    for tid in token_ids:
        piece = tokenizer.id_to_token(tid)
        if piece is None:
            # Should not happen for valid IDs, but guard defensively.
            pieces.append("[UNK]")
        elif piece not in _SKIP_TOKENS:
            pieces.append(piece)

    text = "".join(pieces)

    # Restore whitespace characters — order does not matter here because the
    # markers are distinct strings with no overlap.
    text = text.replace("<SPACE>",   " ")
    text = text.replace("<NEWLINE>", "\n")
    text = text.replace("<TAB>",     "\t")
    return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def has_unk(tokenizer, token_ids: list) -> bool:
    """Return True if any token ID maps to [UNK].

    Use this before asserting exact round-trip equality, e.g.:

        decoded = custom_decode(tok, ids)
        if not has_unk(tok, ids):
            assert decoded == normalised_original
    """
    unk_id = tokenizer.get_vocab().get("[UNK]")
    if unk_id is None:
        return False
    return unk_id in token_ids


def normalise_text(text: str) -> str:
    """Apply the same CRLF normalisation as preprocess_text, without marker substitution.

    Useful for computing the expected decoded string when asserting round-trips:

        expected = normalise_text(raw_text)
        decoded  = custom_decode(tok, ids)
        assert decoded == expected  # only when no UNKs
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")
