import csv
import tempfile
import unittest
from pathlib import Path

from ETL import run_etl as etl


RAW_FIELDNAMES = [
    "Release Year",
    "Title",
    "Origin/Ethnicity",
    "Director",
    "Cast",
    "Genre",
    "Wiki Page",
    "Plot",
]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class BuildGenreWordIndexTests(unittest.TestCase):
    def test_extract_genre_words_applies_requested_normalization(self) -> None:
        self.assertEqual(
            etl.extract_genre_words("Sci-Fi science-fiction, Action/Comedy"),
            ["scifi", "scifi", "action", "comedy"],
        )

    def test_write_movies_with_line_no_writes_data_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            input_csv = root_dir / "movies.csv"
            movies_out = root_dir / "data" / "movies.csv"

            write_csv(
                input_csv,
                RAW_FIELDNAMES,
                [
                    {
                        "Release Year": "1901",
                        "Title": "Kansas Saloon Smashers",
                        "Origin/Ethnicity": "American",
                        "Director": "Unknown",
                        "Cast": "",
                        "Genre": "unknown",
                        "Wiki Page": "https://example.com/1",
                        "Plot": "Plot 1",
                    },
                    {
                        "Release Year": "1902",
                        "Title": "Movie Two",
                        "Origin/Ethnicity": "American",
                        "Director": "Unknown",
                        "Cast": "",
                        "Genre": "Sci-Fi comedy",
                        "Wiki Page": "https://example.com/2",
                        "Plot": "Plot 2",
                    },
                ],
            )

            row_count = etl.write_movies_with_line_no(input_path=input_csv, movies_out=movies_out)

            self.assertEqual(row_count, 2)
            self.assertEqual(
                read_csv(movies_out),
                [
                    {
                        "Release Year": "1901",
                        "Title": "Kansas Saloon Smashers",
                        "Origin/Ethnicity": "American",
                        "Director": "Unknown",
                        "Cast": "",
                        "Genre": "unknown",
                        "Wiki Page": "https://example.com/1",
                        "Plot": "Plot 1",
                        "line_no": "1",
                    },
                    {
                        "Release Year": "1902",
                        "Title": "Movie Two",
                        "Origin/Ethnicity": "American",
                        "Director": "Unknown",
                        "Cast": "",
                        "Genre": "Sci-Fi comedy",
                        "Wiki Page": "https://example.com/2",
                        "Plot": "Plot 2",
                        "line_no": "2",
                    },
                ],
            )

    def test_build_outputs_writes_words_and_word_id_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            input_csv = root_dir / "movies.csv"
            words_out = root_dir / "genre_words.csv"
            mapping_out = root_dir / "genre_word_mapping.csv"

            write_csv(
                input_csv,
                RAW_FIELDNAMES,
                [
                    {
                        "Release Year": "1901",
                        "Title": "Kansas Saloon Smashers",
                        "Origin/Ethnicity": "American",
                        "Director": "Unknown",
                        "Cast": "",
                        "Genre": "unknown",
                        "Wiki Page": "https://example.com/1",
                        "Plot": "Plot 1",
                    },
                    {
                        "Release Year": "1902",
                        "Title": "Movie Two",
                        "Origin/Ethnicity": "American",
                        "Director": "Unknown",
                        "Cast": "",
                        "Genre": "Sci-Fi comedy",
                        "Wiki Page": "https://example.com/2",
                        "Plot": "Plot 2",
                    },
                    {
                        "Release Year": "1903",
                        "Title": "Movie Three",
                        "Origin/Ethnicity": "American",
                        "Director": "Unknown",
                        "Cast": "",
                        "Genre": "science-fiction, drama",
                        "Wiki Page": "https://example.com/3",
                        "Plot": "Plot 3",
                    },
                ],
            )

            line_count, mapping_count, word_count = etl.build_genre_outputs(
                input_path=input_csv,
                words_out=words_out,
                mapping_out=mapping_out,
            )

            self.assertEqual(line_count, 3)
            self.assertEqual(mapping_count, 5)
            self.assertEqual(word_count, 4)
            self.assertEqual(
                read_csv(words_out),
                [
                    {"genre_id": "1", "word": "unknown", "embedding": "", "pos": "", "frequency": "1"},
                    {"genre_id": "2", "word": "scifi", "embedding": "", "pos": "", "frequency": "2"},
                    {"genre_id": "3", "word": "comedy", "embedding": "", "pos": "", "frequency": "1"},
                    {"genre_id": "4", "word": "drama", "embedding": "", "pos": "", "frequency": "1"},
                ],
            )
            self.assertEqual(
                read_csv(mapping_out),
                [
                    {"line_no": "1", "word_id": "1", "word_position": "1", "length": "1"},
                    {"line_no": "2", "word_id": "2", "word_position": "1", "length": "2"},
                    {"line_no": "2", "word_id": "3", "word_position": "2", "length": "2"},
                    {"line_no": "3", "word_id": "2", "word_position": "1", "length": "2"},
                    {"line_no": "3", "word_id": "4", "word_position": "2", "length": "2"},
                ],
            )

    def test_build_outputs_preserves_repeated_word_occurrences(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            input_csv = root_dir / "movies.csv"
            words_out = root_dir / "genre_words.csv"
            mapping_out = root_dir / "genre_word_mapping.csv"

            write_csv(
                input_csv,
                RAW_FIELDNAMES,
                [
                    {
                        "Release Year": "1901",
                        "Title": "Repeat Movie",
                        "Origin/Ethnicity": "American",
                        "Director": "Unknown",
                        "Cast": "",
                        "Genre": "action action comedy",
                        "Wiki Page": "https://example.com/1",
                        "Plot": "Plot 1",
                    }
                ],
            )

            etl.build_genre_outputs(
                input_path=input_csv,
                words_out=words_out,
                mapping_out=mapping_out,
            )

            self.assertEqual(
                read_csv(mapping_out),
                [
                    {"line_no": "1", "word_id": "1", "word_position": "1", "length": "3"},
                    {"line_no": "1", "word_id": "1", "word_position": "2", "length": "3"},
                    {"line_no": "1", "word_id": "2", "word_position": "3", "length": "3"},
                ],
            )
            self.assertEqual(
                read_csv(words_out),
                [
                    {"genre_id": "1", "word": "action", "embedding": "", "pos": "", "frequency": "2"},
                    {"genre_id": "2", "word": "comedy", "embedding": "", "pos": "", "frequency": "1"},
                ],
            )

    def test_build_outputs_requires_genre_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            input_csv = root_dir / "movies.csv"
            words_out = root_dir / "genre_words.csv"
            mapping_out = root_dir / "genre_word_mapping.csv"

            write_csv(
                input_csv,
                [field for field in RAW_FIELDNAMES if field != "Genre"],
                [
                    {
                        "Release Year": "1901",
                        "Title": "No Genre",
                        "Origin/Ethnicity": "American",
                        "Director": "Unknown",
                        "Cast": "",
                        "Wiki Page": "https://example.com/1",
                        "Plot": "Plot 1",
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "Genre column"):
                etl.build_genre_outputs(
                    input_path=input_csv,
                    words_out=words_out,
                    mapping_out=mapping_out,
                )


if __name__ == "__main__":
    unittest.main()
