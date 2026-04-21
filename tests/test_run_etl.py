import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ETL import run_etl


class RunEtlTests(unittest.TestCase):
    def test_run_pipeline_runs_required_steps_without_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            movies_out = root_dir / "movies.csv"
            genre_out = root_dir / "genre.csv"
            mapping_out = root_dir / "genre_mapping.csv"
            call_order: list[str] = []

            with (
                patch.object(run_etl, "write_movies_csv", side_effect=lambda **kwargs: call_order.append("movies") or 3),
                patch.object(
                    run_etl,
                    "build_genre_outputs",
                    side_effect=lambda **kwargs: call_order.append("genre") or (3, 5, 4),
                ),
                patch.object(
                    run_etl,
                    "annotate_genre_outputs",
                    side_effect=lambda **kwargs: call_order.append("pos") or 4,
                ),
                patch.object(run_etl, "read_genre_rows") as read_genre_rows,
                patch.object(run_etl, "get_openai_client") as get_openai_client,
                patch.object(run_etl, "populate_embeddings") as populate_embeddings,
                patch.object(run_etl, "write_genre_rows") as write_genre_rows,
            ):
                run_etl.run_pipeline(
                    input_path=root_dir / "raw.csv",
                    movies_out=movies_out,
                    genre_out=genre_out,
                    mapping_out=mapping_out,
                    with_embeddings=False,
                )

            self.assertEqual(call_order, ["movies", "genre", "pos"])
            read_genre_rows.assert_not_called()
            get_openai_client.assert_not_called()
            populate_embeddings.assert_not_called()
            write_genre_rows.assert_not_called()

    def test_run_pipeline_optionally_runs_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            movies_out = root_dir / "movies.csv"
            genre_out = root_dir / "genre.csv"
            mapping_out = root_dir / "genre_mapping.csv"
            rows = [{"genre_id": "1", "word": "romantic", "embedding": "", "pos": "ADJ"}]
            updated_rows = [{"genre_id": "1", "word": "romantic", "embedding": "[0.1]", "pos": "ADJ"}]

            with (
                patch.object(run_etl, "write_movies_csv", return_value=3),
                patch.object(run_etl, "build_genre_outputs", return_value=(3, 5, 4)),
                patch.object(run_etl, "annotate_genre_outputs", return_value=4),
                patch.object(run_etl, "read_genre_rows", return_value=rows),
                patch.object(run_etl, "get_openai_client", return_value=object()) as get_openai_client,
                patch.object(run_etl, "populate_embeddings", return_value=updated_rows) as populate_embeddings,
                patch.object(run_etl, "write_genre_rows") as write_genre_rows,
            ):
                run_etl.run_pipeline(
                    input_path=root_dir / "raw.csv",
                    movies_out=movies_out,
                    genre_out=genre_out,
                    mapping_out=mapping_out,
                    with_embeddings=True,
                )

            get_openai_client.assert_called_once()
            populate_embeddings.assert_called_once_with(rows, get_openai_client.return_value)
            write_genre_rows.assert_called_once_with(genre_out, updated_rows)


if __name__ == "__main__":
    unittest.main()
