import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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


class GenerateGenreEmbeddingTests(unittest.TestCase):
    def test_build_genre_rows_from_movies_creates_canonical_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            input_csv = root_dir / "movies.csv"

            write_csv(
                input_csv,
                RAW_FIELDNAMES,
                [
                    {
                        "Release Year": "1901",
                        "Title": "Movie One",
                        "Origin/Ethnicity": "American",
                        "Director": "Unknown",
                        "Cast": "",
                        "Genre": "science fiction comedy",
                        "Wiki Page": "https://example.com/1",
                        "Plot": "Plot 1",
                    }
                ],
            )

            rows = etl.build_genre_rows_from_movies(input_csv)

            self.assertEqual(
                rows,
                [
                    {"genre_id": "1", "word": "sci", "embedding": "", "pos": "", "frequency": "1"},
                    {"genre_id": "2", "word": "scif", "embedding": "", "pos": "", "frequency": "1"},
                    {"genre_id": "3", "word": "scifi", "embedding": "", "pos": "", "frequency": "1"},
                    {"genre_id": "4", "word": "com", "embedding": "", "pos": "", "frequency": "1"},
                    {"genre_id": "5", "word": "come", "embedding": "", "pos": "", "frequency": "1"},
                    {"genre_id": "6", "word": "comed", "embedding": "", "pos": "", "frequency": "1"},
                    {"genre_id": "7", "word": "comedy", "embedding": "", "pos": "", "frequency": "1"},
                ],
            )

    def test_populate_embeddings_preserves_existing_pos(self) -> None:
        rows = [
            {"genre_id": "1", "word": "romantic", "embedding": "", "pos": "ADJ", "frequency": "4"},
            {"genre_id": "2", "word": "comedy", "embedding": "", "pos": "NOUN", "frequency": "7"},
        ]
        fake_client = SimpleNamespace(
            embeddings=SimpleNamespace(
                create=lambda model, input: SimpleNamespace(
                    data=[
                        SimpleNamespace(embedding=[0.1, 0.2]),
                        SimpleNamespace(embedding=[0.3, 0.4]),
                    ]
                )
            )
        )

        updated_rows = etl.populate_embeddings(rows, fake_client)

        self.assertEqual(
            updated_rows,
            [
                {"genre_id": "1", "word": "romantic", "embedding": "[0.1,0.2]", "pos": "ADJ", "frequency": "4"},
                {"genre_id": "2", "word": "comedy", "embedding": "[0.3,0.4]", "pos": "NOUN", "frequency": "7"},
            ],
        )

    def test_write_genre_rows_writes_canonical_genre_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            output_csv = root_dir / "genre.csv"

            etl.write_genre_rows(
                output_csv,
                [
                    {"genre_id": "1", "word": "romantic", "embedding": "[0.1,0.2]", "pos": "ADJ", "frequency": "4"},
                    {"genre_id": "2", "word": "comedy", "embedding": "[0.3,0.4]", "pos": "NOUN", "frequency": "7"},
                ],
            )

            self.assertEqual(
                read_csv(output_csv),
                [
                    {"genre_id": "1", "word": "romantic", "embedding": "[0.1,0.2]", "pos": "ADJ", "frequency": "4"},
                    {"genre_id": "2", "word": "comedy", "embedding": "[0.3,0.4]", "pos": "NOUN", "frequency": "7"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
