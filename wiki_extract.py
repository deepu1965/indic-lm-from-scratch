"""
wiki_extract.py
================

MANUAL DATA COLLECTION stage for a Hindi / Nepali language-model corpus.

Purpose
-------
Stream-parse Wikimedia XML dumps (.xml.bz2) and extract raw article/page
text into JSONL, with NO cleaning, filtering, deduplication, language
detection, quality filtering, or Unicode normalization.

"Extraction" here means only:
    Wikimedia XML dump -> parse XML -> pull out title/id/wikitext
                        -> strip MediaWiki markup down to readable text
                        -> write one JSON record per article, one per line

Stripping wiki markup (templates, [[links]], {{tables}}, <ref> tags, etc.)
is a REQUIRED PART of extracting the article's textual content from the
dump -- it is not "corpus cleaning". No articles are dropped for being
short, low quality, non-Hindi/Nepali, etc. Nothing is deduplicated.
Nothing is Unicode-normalized.

Pip installs
------------
    pip install requests mwparserfromhell

Run
---
    python wiki_extract.py                     # process all configured dumps
    python wiki_extract.py --dump hi_wikipedia  # just Hindi Wikipedia
    python wiki_extract.py --dump hi_wikisource # just Hindi Wikisource
    python wiki_extract.py --dump ne_wikipedia  # just Nepali Wikipedia

Toggle TEST_MODE / COUNT_ARTICLES below before running the full extraction.
"""

import argparse
import bz2
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional
from xml.etree import ElementTree as ET

import requests
import mwparserfromhell


# ============================================================================
# CONFIGURATION SWITCHES
# ============================================================================

# When True: only extract the first TEST_ARTICLES eligible articles from each
# dump, print them for manual verification, then stop. Uses the EXACT SAME
# extraction code path as the full run (see CorpusExtractor.run below).
TEST_MODE = False
TEST_ARTICLES = 5

# When True: before extracting, do a full streaming pass over the dump that
# only counts pages (total pages, main-namespace pages, redirects, eligible
# articles). This requires reading the entire compressed dump once, since
# MediaWiki dumps do not store a page count anywhere in the file -- there is
# no way to get a reliable count without a full scan. It does not store any
# page text and is independent of TEST_MODE/extraction.
COUNT_ARTICLES = False

# How often (in pages) to print a progress update.
PROGRESS_EVERY = 10_000


# ============================================================================
# Dump configuration
# ============================================================================

@dataclass
class DumpConfig:
    """Describes one Wikimedia dump to download and extract."""

    url: str
    language: str      # e.g. "hindi", "nepali"
    project: str        # e.g. "wikipedia", "wikisource", "wikibooks"
    output_root: str    # e.g. "hindi/corpus/manual"
    dump_dir: str = "dumps"

    @property
    def dump_filename(self) -> str:
        return os.path.basename(self.url)

    @property
    def dump_path(self) -> str:
        return os.path.join(self.dump_dir, self.dump_filename)

    @property
    def output_filename(self) -> str:
        suffix = "test" if TEST_MODE else "full"
        return f"{self.language}_{self.project}_{suffix}.jsonl"

    @property
    def output_path(self) -> str:
        return os.path.join(self.output_root, self.output_filename)


DUMP_CONFIGS: Dict[str, DumpConfig] = {
    "hi_wikipedia": DumpConfig(
        url="https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2",
        language="hindi",
        project="wikipedia",
        output_root="hindi/corpus/manual",
    ),
    "hi_wikisource": DumpConfig(
        url="https://dumps.wikimedia.org/hiwikisource/latest/hiwikisource-latest-pages-articles.xml.bz2",
        language="hindi",
        project="wikisource",
        output_root="hindi/corpus/manual",
    ),
    "ne_wikipedia": DumpConfig(
        url="https://dumps.wikimedia.org/newiki/latest/newiki-latest-pages-articles.xml.bz2",
        language="nepali",
        project="wikipedia",
        output_root="nepali/corpus/manual",
    ),
    "ne_wikibooks": DumpConfig(
        url="https://dumps.wikimedia.org/newikibooks/latest/newikibooks-latest-pages-articles.xml.bz2",
        language="nepali",
        project="wikibooks",
        output_root="nepali/corpus/manual",
    ),
}


# ============================================================================
# Downloader
# ============================================================================

class DumpDownloader:
    """Download a Wikimedia dump only if it is not already present locally."""

    CHUNK_SIZE = 1024 * 1024  # 1 MB

    def __init__(self, config: DumpConfig):
        """Initialize the downloader with a dump configuration."""
        self.config = config

    def download(self) -> str:
        """Return the local dump path, downloading it only if necessary."""
        os.makedirs(self.config.dump_dir, exist_ok=True)

        path = self.config.dump_path

        # If the dump already exists, do not download it again.
        if os.path.isfile(path):
            print(f"[SKIP] Dump already exists: {path}")
            return path

        print(f"[DOWNLOAD] Downloading: {self.config.url}")

        with requests.get(
            self.config.url,
            stream=True,
            timeout=60
        ) as response:
            response.raise_for_status()

            with open(path, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=self.CHUNK_SIZE
                ):
                    if chunk:
                        f.write(chunk)

        print(f"[DONE] Download complete: {path}")

        return path


# ============================================================================
# Streaming XML page reader
# ============================================================================

def _strip_ns(tag: str) -> str:
    """Strip the XML namespace prefix from a tag, e.g. '{...}page' -> 'page'."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


class WikiDumpPageReader:
    """Streams raw <page> records out of a bz2-compressed MediaWiki XML dump.

    Uses xml.etree.ElementTree.iterparse directly on the open .bz2 stream so
    the file is never decompressed to disk and never fully loaded into
    memory. Each completed <page> subtree is cleared immediately (and the
    root element periodically cleared) so memory stays bounded no matter how
    large the dump is.

    This class does ONLY structural XML parsing -- no markup stripping, no
    filtering. It yields every <page> element found, main-namespace or not,
    redirect or not; callers decide what to keep.
    """

    def __init__(self, dump_path: str):
        self.dump_path = dump_path

    def pages(self) -> Iterator[Dict[str, Any]]:
        """Yield one dict per <page> element: id, title, ns, redirect, text."""
        with bz2.open(self.dump_path, "rb") as f:
            # We need BOTH start and end events for <revision>: children close
            # (in document order) before their parent does, so a flag that is
            # only set on the 'end' of <revision> would never be True while we
            # are inside it. Using 'start' to raise the flag and 'end' to lower
            # it lets us correctly tell a page-level <id>/<title> apart from a
            # revision/contributor <id>, and correctly find <text> inside
            # <revision>.
            context = ET.iterparse(f, events=("start", "end"))

            page_id = None
            title = None
            ns = None
            is_redirect = False
            text = None
            in_revision = False

            for event, elem in context:
                tag = _strip_ns(elem.tag)

                if tag == "revision":
                    in_revision = (event == "start")
                    continue

                if event != "end":
                    # Only <revision> needs the 'start' event; everything else
                    # is read once its own end tag is reached.
                    continue

                if tag == "title" and not in_revision:
                    title = elem.text or ""
                elif tag == "ns" and not in_revision:
                    ns = elem.text if elem.text is not None else "0"
                elif tag == "id" and not in_revision and page_id is None:
                    # The first <id> under <page> (i.e. before <revision> starts)
                    # is the page id, not a revision/contributor id.
                    page_id = elem.text or ""
                elif tag == "redirect":
                    is_redirect = True
                elif tag == "text" and in_revision:
                    text = elem.text or ""
                elif tag == "page":
                    yield {
                        "id": page_id,
                        "title": title,
                        "ns": ns,
                        "redirect": is_redirect,
                        "text": text or "",
                    }
                    # Reset per-page state.
                    page_id = None
                    title = None
                    ns = None
                    is_redirect = False
                    text = None
                    in_revision = False
                    # Free this page's subtree; keeps memory bounded.
                    elem.clear()
                elif tag == "mediawiki":
                    # Dump fully consumed.
                    elem.clear()


# ============================================================================
# Wikitext -> plain text extraction
# ============================================================================

class WikiTextExtractor:
    """Converts raw MediaWiki wikitext into readable plain text.

    This performs ONLY the structural work needed to recover the article's
    textual content from wiki markup: stripping templates, wikilink/table
    syntax, <ref> tags, etc., via mwparserfromhell. HTML entity decoding
    (e.g. "&amp;" -> "&") is part of correct extraction and is applied; this
    is NOT the same as Unicode normalization (NFC/NFKC), which is
    intentionally NOT performed here.

    Leftover MediaWiki image/file markup fragments (e.g. "300 px|thumb|",
    "thumb|left|") that survive strip_code are cleaned in the downstream
    corpus cleaning pipeline (clean_corpus.py, MediaWikiMarkupCleaner),
    not here, so that the fix can be applied to already-extracted JSONL
    files without re-running extraction.
    """

    @staticmethod
    def extract(wikitext: str) -> str:
        if not wikitext:
            return ""
        parsed = mwparserfromhell.parse(wikitext)
        plain = parsed.strip_code(normalize=True, collapse=False)
        return plain.strip()


def count_words(text: str) -> int:
    """Simple whitespace word count, for REPORTING ONLY (never used to filter)."""
    return len(text.split())


# ============================================================================
# Article counting (optional full scan)
# ============================================================================

class ArticleCounter:
    """Scans an entire dump once to count pages by type. No text is kept.

    A reliable total-article count cannot be obtained without reading the
    whole dump (MediaWiki dumps carry no such count in their header), so
    this always performs a full streaming pass when enabled.
    """

    def __init__(self, dump_path: str, progress_every: int):
        self.dump_path = dump_path
        self.progress_every = progress_every

    def count(self) -> Dict[str, int]:
        reader = WikiDumpPageReader(self.dump_path)
        total_pages = 0
        ns0_pages = 0
        redirects = 0

        print("[COUNT] Scanning dump to count pages (requires a full pass, may take a while)...")
        start = time.time()
        for page in reader.pages():
            total_pages += 1
            if page["ns"] == "0":
                ns0_pages += 1
                if page["redirect"]:
                    redirects += 1
            if total_pages % self.progress_every == 0:
                print(f"[COUNT PROGRESS] pages_scanned={total_pages}")

        eligible = ns0_pages - redirects
        elapsed = time.time() - start
        print(f"[COUNT DONE] elapsed={elapsed:.1f}s")
        print(f"  Total pages found:                     {total_pages}")
        print(f"  Total main-namespace (ns=0) pages:      {ns0_pages}")
        print(f"  Total redirects (within ns=0):          {redirects}")
        print(f"  Total articles eligible for extraction: {eligible}")
        return {
            "total_pages": total_pages,
            "ns0_pages": ns0_pages,
            "redirects": redirects,
            "eligible": eligible,
        }


# ============================================================================
# Corpus extraction (test mode and full mode share this exact code path)
# ============================================================================

class CorpusExtractor:
    """Extract text from Wikipedia or Wikisource dumps into JSONL."""

    def __init__(
        self,
        config: DumpConfig,
        test_mode: bool,
        test_articles: int,
        progress_every: int,
    ):
        """Initialize the corpus extractor."""
        self.config = config
        self.test_mode = test_mode
        self.test_articles = test_articles
        self.progress_every = progress_every

    def is_eligible(self, page):
        """Return whether a page contains corpus text for this project."""
        if self.config.project == "wikisource":
            # Wikisource stores the actual scanned/OCR/proofread text
            # in the Page namespace (250).
            return page["ns"] == "250"

        # Wikipedia uses the main article namespace (0).
        return page["ns"] == "0" and not page["redirect"]

    def run(self, dump_path: str) -> None:
        """Extract eligible pages from the dump and write them as JSONL."""
        os.makedirs(self.config.output_root, exist_ok=True)

        reader = WikiDumpPageReader(dump_path)

        pages_seen = 0
        records_written = 0
        total_words = 0
        total_chars = 0

        start = time.time()

        mode_label = (
            f"TEST MODE (first {self.test_articles} pages)"
            if self.test_mode
            else "FULL RUN"
        )

        print(f"[EXTRACT] Starting extraction -- {mode_label}")
        print(f"[EXTRACT] Output: {self.config.output_path}")

        with open(
            self.config.output_path,
            "w",
            encoding="utf-8"
        ) as out_f:

            for page in reader.pages():
                pages_seen += 1

                if pages_seen % self.progress_every == 0:
                    elapsed = time.time() - start

                    print(
                        f"[PROGRESS] pages_processed={pages_seen} "
                        f"records_written={records_written} "
                        f"words={total_words} "
                        f"chars={total_chars} "
                        f"elapsed={elapsed:.1f}s"
                    )

                # -------------------------------------------------------
                # Select the appropriate namespace.
                # Wikipedia  -> ns=0
                # Wikisource -> ns=250
                # -------------------------------------------------------
                if not self.is_eligible(page):
                    continue

                plain_text = WikiTextExtractor.extract(page["text"])

                words = count_words(plain_text)
                chars = len(plain_text)

                record = {
                    "id": page["id"],
                    "title": page["title"],
                    "text": plain_text,
                    "source": "wikimedia",
                    "language": self.config.language,
                    "project": self.config.project,
                }

                out_f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )

                records_written += 1
                total_words += words
                total_chars += chars

                if self.test_mode:
                    print(
                        f"[TEST] #{records_written} "
                        f"id={page['id']} "
                        f"title={page['title']!r} "
                        f"words={words} "
                        f"chars={chars}"
                    )

                if (
                    self.test_mode
                    and records_written >= self.test_articles
                ):
                    print(
                        f"[TEST MODE] Reached "
                        f"{self.test_articles} extracted pages -- stopping."
                    )
                    break

        elapsed = time.time() - start

        print("\n[SUMMARY]")
        print(f"  Dump:             {dump_path}")
        print(f"  Output:           {self.config.output_path}")
        print(f"  Pages processed:  {pages_seen}")
        print(f"  Records written:  {records_written}")
        print(f"  Total words:      {total_words}")
        print(f"  Total characters: {total_chars}")
        print(f"  Elapsed:          {elapsed:.1f}s")
        print(f"  FINAL TOTAL WORDS EXTRACTED: {total_words}")

# ============================================================================
# Orchestration
# ============================================================================

def process_dump(key: str, config: DumpConfig) -> None:
    print(f"\n{'=' * 78}")
    print(f"Dump: {key}  ({config.language} / {config.project})")
    print(f"URL:  {config.url}")
    print(f"{'=' * 78}")

    dump_path = DumpDownloader(config).download()

    if COUNT_ARTICLES:
        ArticleCounter(dump_path, PROGRESS_EVERY).count()

    CorpusExtractor(config, TEST_MODE, TEST_ARTICLES, PROGRESS_EVERY).run(dump_path)


def main():
    parser = argparse.ArgumentParser(description="Stream-extract Wikimedia dumps to JSONL (manual collection stage).")
    parser.add_argument(
        "--dump",
        choices=list(DUMP_CONFIGS.keys()) + ["all"],
        default="all",
        help="Which dump to process (default: all configured dumps).",
    )
    args = parser.parse_args()

    if args.dump == "all":
        for key, config in DUMP_CONFIGS.items():
            process_dump(key, config)
    else:
        process_dump(args.dump, DUMP_CONFIGS[args.dump])


if __name__ == "__main__":
    main()