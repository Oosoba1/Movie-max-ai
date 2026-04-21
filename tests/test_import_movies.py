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


class ImportMoviesTests(unittest.TestCase):
    def test_write_movies_csv_writes_id_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            raw_movies = root_dir / "movies.csv"
            movies_out = root_dir / "data" / "movies.csv"

            write_csv(
                raw_movies,
                RAW_FIELDNAMES,
                [
                    {
                        "Release Year": "1901",
                        "Title": "Movie One",
                        "Origin/Ethnicity": "American",
                        "Director": "Unknown",
                        "Cast": "",
                        "Genre": "unknown",
                        "Wiki Page": "https://example.com/1",
                        "Plot": "Plot 1",
                    }
                ],
            )

            row_count = etl.write_movies_csv(input_path=raw_movies, output_path=movies_out)

            self.assertEqual(row_count, 1)
            self.assertEqual(
                read_csv(movies_out),
                [
                    {
                        "Release Year": "1901",
                        "Title": "Movie One",
                        "Origin/Ethnicity": "American",
                        "Director": "Unknown",
                        "Cast": "",
                        "Genre": "unknown",
                        "Wiki Page": "https://example.com/1",
                        "Plot": "Plot 1",
                        "id": "1",
                    }
                ],
            )

    def test_write_movies_csv_preserves_existing_line_no_as_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            input_csv = root_dir / "movies.csv"
            output_csv = root_dir / "data" / "movies.csv"

            write_csv(
                input_csv,
                RAW_FIELDNAMES + ["line_no"],
                [
                    {
                        "Release Year": "1901",
                        "Title": "Movie One",
                        "Origin/Ethnicity": "American",
                        "Director": "Unknown",
                        "Cast": "",
                        "Genre": "unknown",
                        "Wiki Page": "https://example.com/1",
                        "Plot": "Plot 1",
                        "line_no": "42",
                    }
                ],
            )

            row_count = etl.write_movies_csv(input_path=input_csv, output_path=output_csv)

            self.assertEqual(row_count, 1)
            self.assertEqual(read_csv(output_csv)[0]["id"], "42")


if __name__ == "__main__":
    unittest.main()
