import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ETL import run_etl as etl


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class AnnotateGenrePartsOfSpeechTests(unittest.TestCase):
    def test_normalize_possible_parts_of_speech_orders_and_deduplicates(self) -> None:
        self.assertEqual(
            etl.normalize_possible_parts_of_speech({"ADV", "NOUN", "ADJ", "NOUN"}),
            ["NOUN", "ADJ", "ADV"],
        )

    def test_possible_parts_of_speech_uses_tagger_fallback_when_wordnet_has_no_match(self) -> None:
        with (
            patch.object(etl, "lookup_wordnet_parts_of_speech", return_value=set()),
            patch.object(etl, "guess_tagged_part_of_speech", return_value="NOUN"),
        ):
            self.assertEqual(etl.possible_parts_of_speech_for_word("scifi"), ["NOUN"])

    def test_build_outputs_writes_possible_parts_of_speech(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            input_csv = root_dir / "genre.csv"
            output_csv = root_dir / "genre.csv.out"

            write_csv(
                input_csv,
                ["genre_id", "word", "embedding", "pos", "frequency"],
                [
                    {"genre_id": "1", "word": "unknown", "embedding": "[1.0]", "pos": "", "frequency": "5"},
                    {"genre_id": "2", "word": "scifi", "embedding": "", "pos": "", "frequency": "2"},
                    {"genre_id": "3", "word": "romantic", "embedding": "", "pos": "", "frequency": "1"},
                ],
            )

            with patch.object(
                etl,
                "possible_parts_of_speech_for_word",
                side_effect=lambda word, contextual_parts_of_speech=None: {
                    "unknown": ["ADJ", "NOUN"],
                    "scifi": ["NOUN"],
                    "romantic": ["ADJ"],
                }[word],
            ):
                row_count = etl.annotate_genre_outputs(input_path=input_csv, output_path=output_csv)

            self.assertEqual(row_count, 3)
            self.assertEqual(
                read_csv(output_csv),
                [
                    {"genre_id": "1", "word": "unknown", "embedding": "[1.0]", "pos": "ADJ|NOUN", "frequency": "5"},
                    {"genre_id": "2", "word": "scifi", "embedding": "", "pos": "NOUN", "frequency": "2"},
                    {"genre_id": "3", "word": "romantic", "embedding": "", "pos": "ADJ", "frequency": "1"},
                ],
            )

    def test_build_outputs_uses_mapping_sequence_context_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            input_csv = root_dir / "genre.csv"
            mapping_csv = root_dir / "genre_mapping.csv"
            output_csv = root_dir / "genre.csv.out"

            write_csv(
                input_csv,
                ["genre_id", "word", "embedding", "pos", "frequency"],
                [
                    {"genre_id": "1", "word": "romantic", "embedding": "", "pos": "", "frequency": "3"},
                    {"genre_id": "2", "word": "comedy", "embedding": "", "pos": "", "frequency": "8"},
                ],
            )
            write_csv(
                mapping_csv,
                ["line_no", "word_id", "word_position"],
                [
                    {"line_no": "10", "word_id": "1", "word_position": "1"},
                    {"line_no": "10", "word_id": "2", "word_position": "2"},
                ],
            )

            with patch.object(
                etl,
                "infer_parts_of_speech_for_sequence",
                return_value=["ADJ", "NOUN"],
            ):
                row_count = etl.annotate_genre_outputs(
                    input_path=input_csv,
                    output_path=output_csv,
                    mapping_path=mapping_csv,
                )

            self.assertEqual(row_count, 2)
            self.assertEqual(
                read_csv(output_csv),
                [
                    {"genre_id": "1", "word": "romantic", "embedding": "", "pos": "ADJ", "frequency": "3"},
                    {"genre_id": "2", "word": "comedy", "embedding": "", "pos": "NOUN", "frequency": "8"},
                ],
            )

    def test_build_outputs_supports_mapping_files_with_inline_words(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            input_csv = root_dir / "genre.csv"
            mapping_csv = root_dir / "genre_mapping.csv"
            output_csv = root_dir / "genre.csv.out"

            write_csv(
                input_csv,
                ["genre_id", "word", "embedding", "pos", "frequency"],
                [
                    {"genre_id": "1", "word": "historical", "embedding": "", "pos": "", "frequency": "6"},
                    {"genre_id": "2", "word": "drama", "embedding": "", "pos": "", "frequency": "9"},
                ],
            )
            write_csv(
                mapping_csv,
                ["line_no", "word", "word_order"],
                [
                    {"line_no": "20", "word": "historical", "word_order": "1"},
                    {"line_no": "20", "word": "drama", "word_order": "2"},
                ],
            )

            with patch.object(
                etl,
                "infer_parts_of_speech_for_sequence",
                return_value=["ADJ", "NOUN"],
            ):
                row_count = etl.annotate_genre_outputs(
                    input_path=input_csv,
                    output_path=output_csv,
                    mapping_path=mapping_csv,
                )

            self.assertEqual(row_count, 2)
            self.assertEqual(
                read_csv(output_csv),
                [
                    {"genre_id": "1", "word": "historical", "embedding": "", "pos": "ADJ", "frequency": "6"},
                    {"genre_id": "2", "word": "drama", "embedding": "", "pos": "NOUN", "frequency": "9"},
                ],
            )

    def test_build_outputs_allows_in_place_updates_for_genre_csv_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            input_output_csv = root_dir / "genre.csv"

            write_csv(
                input_output_csv,
                ["genre_id", "word", "embedding", "pos", "frequency"],
                [
                    {"genre_id": "1", "word": "romantic", "embedding": "[1.0]", "pos": "NOUN|ADJ", "frequency": "4"},
                    {"genre_id": "2", "word": "comedy", "embedding": "", "pos": "NOUN", "frequency": "7"},
                ],
            )

            with patch.object(
                etl,
                "possible_parts_of_speech_for_word",
                side_effect=lambda word, contextual_parts_of_speech=None: {
                    "romantic": ["ADJ"],
                    "comedy": ["NOUN"],
                }[word],
            ):
                row_count = etl.annotate_genre_outputs(
                    input_path=input_output_csv,
                    output_path=input_output_csv,
                )

            self.assertEqual(row_count, 2)
            self.assertEqual(
                read_csv(input_output_csv),
                [
                    {"genre_id": "1", "word": "romantic", "embedding": "[1.0]", "pos": "ADJ", "frequency": "4"},
                    {"genre_id": "2", "word": "comedy", "embedding": "", "pos": "NOUN", "frequency": "7"},
                ],
            )

    def test_build_outputs_requires_id_and_word_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            input_csv = root_dir / "genre.csv"
            output_csv = root_dir / "genre.csv.out"

            write_csv(
                input_csv,
                ["genre_id", "token"],
                [{"genre_id": "1", "token": "unknown"}],
            )

            with self.assertRaisesRegex(ValueError, "genre_id/id and word columns"):
                etl.annotate_genre_outputs(input_path=input_csv, output_path=output_csv)


if __name__ == "__main__":
    unittest.main()
