"""
Download Sangraha Verified Hindi data and split each file into:
    hindi/corpus/public/web/
    hindi/corpus/public/pdf/
    hindi/corpus/public/speech/

Only the first MAX_FILES files in the repository path are processed.

Already processed files are skipped.
The original downloaded Parquet file is deleted after splitting.

Processing is done row-group by row-group to avoid loading an entire
large Parquet file into RAM.

Each output row additionally contains:
    word_count
    char_count
"""

from pathlib import Path
import os

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import list_repo_files, hf_hub_download, login


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# Change this value later to process more files.
MAX_FILES = 10


# -------------------------------------------------------------------
# Hugging Face authentication
# -------------------------------------------------------------------

token = os.environ.get("HF_TOKEN")

if token:
    login(token=token)


class SangrahaDownloader:
    """Download and split Sangraha Verified Hindi Parquet files."""

    REPO_ID = "ai4bharat/sangraha"
    REPO_PATH = "verified/hin"

    def __init__(self, output_dir="hindi/corpus/public"):
        """Initialize the output directories."""
        self.output_dir = Path(output_dir)

        self.web_dir = self.output_dir / "web"
        self.pdf_dir = self.output_dir / "pdf"
        self.speech_dir = self.output_dir / "speech"

        self.web_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.speech_dir.mkdir(parents=True, exist_ok=True)

    def get_files(self):
        """Return Parquet files from the repository in repository order."""
        files = list_repo_files(
            repo_id=self.REPO_ID,
            repo_type="dataset"
        )

        return [
            f
            for f in files
            if f.startswith(self.REPO_PATH + "/")
            and f.endswith(".parquet")
        ]

    def is_processed(self, filename):
        """Check whether all three split output files already exist."""
        web_file = self.web_dir / filename
        pdf_file = self.pdf_dir / filename
        speech_file = self.speech_dir / filename

        return (
            web_file.exists()
            and pdf_file.exists()
            and speech_file.exists()
        )

    @staticmethod
    def add_counts(table):
        """
        Add word_count and char_count columns to a PyArrow table.

        Counts are calculated independently for every row.
        """
        text_column = table.column("text")

        # Convert null values to empty strings.
        text_column = pa.compute.fill_null(text_column, "")

        # Convert to Python strings for reliable Unicode handling.
        texts = text_column.to_pylist()

        word_counts = []
        char_counts = []

        for text in texts:
            if not isinstance(text, str):
                text = str(text)

            word_counts.append(len(text.split()))
            char_counts.append(len(text))

        table = table.append_column(
            "word_count",
            pa.array(word_counts, type=pa.int64())
        )

        table = table.append_column(
            "char_count",
            pa.array(char_counts, type=pa.int64())
        )

        return table

    def process_file(self, repo_file):
        """
        Download and process one Parquet file row-group by row-group.

        The file is split into web, pdf, and speech outputs.
        Word/character counts are added to every output row.
        """
        filename = Path(repo_file).name

        if self.is_processed(filename):
            print(f"\n[SKIP] {filename} - already processed.")
            return None

        print(f"\n{'=' * 70}")
        print(f"Processing: {filename}")
        print(f"{'=' * 70}")

        # ---------------------------------------------------------------
        # Download one original Parquet file.
        # ---------------------------------------------------------------

        local_file = hf_hub_download(
            repo_id=self.REPO_ID,
            filename=repo_file,
            repo_type="dataset"
        )

        writers = {}

        totals = {
            "web_rows": 0,
            "web_words": 0,
            "web_chars": 0,

            "pdf_rows": 0,
            "pdf_words": 0,
            "pdf_chars": 0,

            "speech_rows": 0,
            "speech_words": 0,
            "speech_chars": 0,
        }

        try:
            # Open Parquet without loading the complete file.
            parquet_file = pq.ParquetFile(local_file)

            total_row_groups = parquet_file.num_row_groups

            print(f"Row groups: {total_row_groups}")

            # -----------------------------------------------------------
            # Process one row group at a time.
            # -----------------------------------------------------------

            for row_group_index in range(total_row_groups):

                print(
                    f"\n  Row group "
                    f"{row_group_index + 1}/{total_row_groups}"
                )

                table = parquet_file.read_row_group(
                    row_group_index
                )

                # -------------------------------------------------------
                # Get the type column.
                # -------------------------------------------------------

                type_values = table.column("type").to_pylist()

                # -------------------------------------------------------
                # Process each source type.
                # -------------------------------------------------------

                for source_type in ("web", "pdf", "speech"):

                    # Find rows belonging to this type.
                    indices = [
                        i
                        for i, value in enumerate(type_values)
                        if str(value).lower() == source_type
                    ]

                    if not indices:
                        continue

                    # Select only this source type.
                    type_table = table.take(
                        pa.array(indices, type=pa.int64())
                    )

                    # Add word_count and char_count.
                    type_table = self.add_counts(type_table)

                    # ---------------------------------------------------
                    # Update cumulative statistics.
                    # ---------------------------------------------------

                    rows = type_table.num_rows

                    words = sum(
                        type_table
                        .column("word_count")
                        .to_pylist()
                    )

                    chars = sum(
                        type_table
                        .column("char_count")
                        .to_pylist()
                    )

                    totals[f"{source_type}_rows"] += rows
                    totals[f"{source_type}_words"] += words
                    totals[f"{source_type}_chars"] += chars

                    # ---------------------------------------------------
                    # Select output path.
                    # ---------------------------------------------------

                    if source_type == "web":
                        output_file = self.web_dir / filename

                    elif source_type == "pdf":
                        output_file = self.pdf_dir / filename

                    else:
                        output_file = self.speech_dir / filename

                    # ---------------------------------------------------
                    # Create Parquet writer if this is the first batch.
                    # ---------------------------------------------------

                    if source_type not in writers:

                        writers[source_type] = pq.ParquetWriter(
                            output_file,
                            type_table.schema
                        )

                    # Write this row group.
                    writers[source_type].write_table(type_table)

                # Release references before reading the next row group.
                del table

                print(
                    f"    cumulative:"
                    f" web={totals['web_words']:,} words,"
                    f" pdf={totals['pdf_words']:,} words,"
                    f" speech={totals['speech_words']:,} words"
                )

            # -----------------------------------------------------------
            # Close all output writers.
            # -----------------------------------------------------------

            for writer in writers.values():
                writer.close()

            writers.clear()

            # -----------------------------------------------------------
            # Print file totals.
            # -----------------------------------------------------------

            print(f"\nFILE TOTAL: {filename}")

            print(
                f"  Web:"
                f"    {totals['web_rows']:,} rows | "
                f"{totals['web_words']:,} words | "
                f"{totals['web_chars']:,} chars"
            )

            print(
                f"  PDF:"
                f"    {totals['pdf_rows']:,} rows | "
                f"{totals['pdf_words']:,} words | "
                f"{totals['pdf_chars']:,} chars"
            )

            print(
                f"  Speech:"
                f" {totals['speech_rows']:,} rows | "
                f"{totals['speech_words']:,} words | "
                f"{totals['speech_chars']:,} chars"
            )

            return totals

        finally:
            # Close writers if an exception occurs.
            for writer in writers.values():
                writer.close()

            # Delete the downloaded original.
            Path(local_file).unlink(missing_ok=True)

            print(
                f"\n[CLEANUP] Removed downloaded original: "
                f"{local_file}"
            )

    def run(self):
        """Process at most MAX_FILES repository files and report totals."""
        files = self.get_files()

        print(f"Total Parquet files in repository: {len(files)}")
        print(f"Maximum files to process: {MAX_FILES}")

        # ---------------------------------------------------------------
        # Select only the first MAX_FILES files in repository order.
        # ---------------------------------------------------------------

        files_to_process = files[:MAX_FILES]

        print("\nFiles selected:")

        for file in files_to_process:
            print(f"  {Path(file).name}")

        # ---------------------------------------------------------------
        # Overall cumulative totals.
        # ---------------------------------------------------------------

        totals = {
            "web_rows": 0,
            "web_words": 0,
            "web_chars": 0,

            "pdf_rows": 0,
            "pdf_words": 0,
            "pdf_chars": 0,

            "speech_rows": 0,
            "speech_words": 0,
            "speech_chars": 0,
        }

        print("\nStarting processing...")

        # ---------------------------------------------------------------
        # Process files one by one.
        # ---------------------------------------------------------------

        for repo_file in files_to_process:

            stats = self.process_file(repo_file)

            # Already processed.
            if stats is None:
                continue

            for key in totals:
                totals[key] += stats[key]

            # -----------------------------------------------------------
            # Print cumulative totals after every file.
            # -----------------------------------------------------------

            print("\nCUMULATIVE TOTALS")

            print(
                f"  Web:"
                f"    {totals['web_rows']:,} rows | "
                f"{totals['web_words']:,} words | "
                f"{totals['web_chars']:,} chars"
            )

            print(
                f"  PDF:"
                f"    {totals['pdf_rows']:,} rows | "
                f"{totals['pdf_words']:,} words | "
                f"{totals['pdf_chars']:,} chars"
            )

            print(
                f"  Speech:"
                f" {totals['speech_rows']:,} rows | "
                f"{totals['speech_words']:,} words | "
                f"{totals['speech_chars']:,} chars"
            )

        # ---------------------------------------------------------------
        # Final totals.
        # ---------------------------------------------------------------

        print("\n" + "=" * 70)
        print("FINAL TOTALS")
        print("=" * 70)

        print(
            f"Web:"
            f"    {totals['web_rows']:,} rows | "
            f"{totals['web_words']:,} words | "
            f"{totals['web_chars']:,} chars"
        )

        print(
            f"PDF:"
            f"    {totals['pdf_rows']:,} rows | "
            f"{totals['pdf_words']:,} words | "
            f"{totals['pdf_chars']:,} chars"
        )

        print(
            f"Speech:"
            f" {totals['speech_rows']:,} rows | "
            f"{totals['speech_words']:,} words | "
            f"{totals['speech_chars']:,} chars"
        )

        print("=" * 70)
        print("\nDone.")


if __name__ == "__main__":
    downloader = SangrahaDownloader()
    downloader.run()
