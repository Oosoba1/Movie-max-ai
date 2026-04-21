import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


class BuildGenreMappingTests(unittest.TestCase):
    def test_build_outputs_prefers_raw_source_over_existing_generated_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            data_dir = root_dir / "data"
            archive_path = data_dir / "data.tgz"
            raw_movies = root_dir / "movies.csv"
            movies_out = data_dir / "movies.csv"
            genre_out = data_dir / "genre.csv"
            mapping_out = data_dir / "genre_mapping.csv"

            write_csv(
                raw_movies,
                RAW_FIELDNAMES,
                [
                    {
                        "Release Year": "1999",
                        "Title": "Raw Source Movie",
                        "Origin/Ethnicity": "American",
                        "Director": "Director",
                        "Cast": "Cast",
                        "Genre": "science fiction",
                        "Wiki Page": "https://example.com/raw",
                        "Plot": "Plot from the raw source.",
                    }
                ],
            )
            write_csv(
                movies_out,
                RAW_FIELDNAMES + ["id"],
                [
                    {
                        "Release Year": "2001",
                        "Title": "Bad Generated Output",
                        "Origin/Ethnicity": "American",
                        "Director": "Wrong Director",
                        "Cast": "Wrong Cast",
                        "Genre": "bad data",
                        "Wiki Page": "https://example.com/bad",
                        "Plot": "This row should not be reused as source input.",
                        "id": "999",
                    }
                ],
            )

            with (
                patch.object(etl, "ROOT_DIR", root_dir),
                patch.object(etl, "DATA_DIR", data_dir),
                patch.object(etl, "ARCHIVE_PATH", archive_path),
            ):
                movie_count, mapping_count, genre_count = etl.build_source_outputs(
                    input_path=None,
                    movies_out=movies_out,
                    genre_out=genre_out,
                    mapping_out=mapping_out,
                )

            output_rows = read_csv(movies_out)
            mapping_rows = read_csv(mapping_out)
            genre_rows = read_csv(genre_out)

            self.assertEqual(movie_count, 1)
            self.assertEqual(mapping_count, 3)
            self.assertEqual(genre_count, 3)
            self.assertEqual(output_rows[0]["Title"], "Raw Source Movie")
            self.assertEqual(output_rows[0]["id"], "1")
            self.assertEqual(
                [(row["line_no"], row["word"], row["word_order"]) for row in mapping_rows],
                [("1", "sci", "1"), ("1", "scif", "1"), ("1", "scifi", "1")],
            )
            self.assertEqual(
                genre_rows,
                [
                    {"genre_id": "1", "word": "sci", "embedding": "", "pos": "", "frequency": ""},
                    {"genre_id": "2", "word": "scif", "embedding": "", "pos": "", "frequency": ""},
                    {"genre_id": "3", "word": "scifi", "embedding": "", "pos": "", "frequency": ""},
                ],
            )

    def test_build_outputs_rejects_same_input_and_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            data_dir = root_dir / "data"
            archive_path = data_dir / "data.tgz"
            movies_out = data_dir / "movies.csv"
            genre_out = data_dir / "genre.csv"
            mapping_out = data_dir / "genre_mapping.csv"

            write_csv(
                movies_out,
                RAW_FIELDNAMES + ["id"],
                [
                    {
                        "Release Year": "2001",
                        "Title": "Already Generated",
                        "Origin/Ethnicity": "American",
                        "Director": "Director",
                        "Cast": "Cast",
                        "Genre": "comedy",
                        "Wiki Page": "https://example.com/movie",
                        "Plot": "Plot.",
                        "id": "7",
                    }
                ],
            )

            with (
                patch.object(etl, "ROOT_DIR", root_dir),
                patch.object(etl, "DATA_DIR", data_dir),
                patch.object(etl, "ARCHIVE_PATH", archive_path),
            ):
                with self.assertRaisesRegex(ValueError, "same file"):
                    etl.build_source_outputs(
                        input_path=movies_out,
                        movies_out=movies_out,
                        genre_out=genre_out,
                        mapping_out=mapping_out,
                    )

    def test_build_outputs_escapes_special_characters_from_external_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as external_tmpdir:
            root_dir = Path(tmpdir)
            data_dir = root_dir / "data"
            archive_path = data_dir / "data.tgz"
            external_input = Path(external_tmpdir) / "movies source.csv"
            movies_out = data_dir / "movies.csv"
            genre_out = data_dir / "genre.csv"
            mapping_out = data_dir / "genre_mapping.csv"

            write_csv(
                external_input,
                RAW_FIELDNAMES,
                [
                    {
                        "Release Year": "2005",
                        "Title": 'Movie, "Quoted"',
                        "Origin/Ethnicity": "American",
                        "Director": 'Director "Name"',
                        "Cast": "Lead, Support",
                        "Genre": "science fiction",
                        "Wiki Page": "https://example.com/movie",
                        "Plot": 'Line one.\nLine two says "hello", then ends.',
                    }
                ],
            )

            with (
                patch.object(etl, "ROOT_DIR", root_dir),
                patch.object(etl, "DATA_DIR", data_dir),
                patch.object(etl, "ARCHIVE_PATH", archive_path),
            ):
                etl.build_source_outputs(
                    input_path=external_input,
                    movies_out=movies_out,
                    genre_out=genre_out,
                    mapping_out=mapping_out,
                )

            output_text = movies_out.read_text(encoding="utf-8")
            output_rows = read_csv(movies_out)

            self.assertEqual(len(output_text.splitlines()), 2)
            self.assertIn('"Movie, ""Quoted"""', output_text)
            self.assertIn(r'Line one.\nLine two says ""hello"", then ends.', output_text)
            self.assertEqual(output_rows[0]["Title"], 'Movie, "Quoted"')
            self.assertEqual(output_rows[0]["Plot"], r'Line one.\nLine two says "hello", then ends.')
            self.assertEqual(output_rows[0]["id"], "1")

    def test_extract_genre_words_adds_partial_prefixes_for_each_word(self) -> None:
        self.assertEqual(
            etl.extract_source_genre_word_parts("Action Comedy"),
            [
                (1, "act"),
                (1, "acti"),
                (1, "actio"),
                (1, "action"),
                (2, "com"),
                (2, "come"),
                (2, "comed"),
                (2, "comedy"),
            ],
        )
        self.assertEqual(
            etl.extract_source_genre_words("Action Comedy"),
            ["act", "acti", "actio", "action", "com", "come", "comed", "comedy"],
        )


if __name__ == "__main__":
    unittest.main()
