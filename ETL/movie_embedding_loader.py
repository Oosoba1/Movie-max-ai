import argparse
import csv
import os
import time
from pathlib import Path
from typing import Iterable

from openai import BadRequestError, OpenAI

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "50"))
MAX_MOVIES = int(os.getenv("EMBED_MAX_MOVIES", "10"))
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_CSV = ROOT_DIR / "data" / "movies.csv"
DEFAULT_OUTPUT_CSV = ROOT_DIR / "data" / "movie_embedding.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate movie embeddings from a movies CSV file."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of movies to embed. Use 0 or a negative value for all rows.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=int(os.getenv("EMBED_MAX_CHARS", "24000")),
        help="Trim input text to this many characters before embedding.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of movies to embed per batch request.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing output and rebuild embeddings from scratch.",
    )
    return parser.parse_args()


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def build_movie_text(row: dict[str, str]) -> str:
    title = (row.get("Title") or "").strip()
    genre = (row.get("Genre") or "").strip()
    plot = (row.get("Plot") or "").strip()
    return " ".join(part for part in [title, genre, plot] if part)


def fetch_movies(input_csv: Path) -> list[tuple[str, str]]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    movies: list[tuple[str, str]] = []
    with input_csv.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError("The input CSV is missing a header row.")

        if "Title" not in reader.fieldnames:
            raise ValueError('The input CSV must include a "Title" column.')

        for row_number, row in enumerate(reader, start=1):
            title = (row.get("Title") or "").strip()
            if not title:
                continue

            movie_text = build_movie_text(row)
            movie_id = (row.get("id") or "").strip() or str(row_number)
            movies.append((movie_id, movie_text))

    return movies


def chunked(items: list[tuple[str, str]], size: int) -> Iterable[list[tuple[str, str]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(value) for value in embedding) + "]"


def movie_text_key(text: str) -> str:
    return " ".join((text or "").split()).lower()


def load_precomputed_state(output_csv: Path) -> tuple[set[str], set[str]]:
    if not output_csv.exists():
        return set(), set()

    with output_csv.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return set(), set()

        ids: set[str] = set()
        text_keys: set[str] = set()

        for row in reader:
            embedding_value = (row.get("embedding") or "").strip()
            if not embedding_value:
                continue

            movie_id = (row.get("movie_id") or "").strip()
            if movie_id:
                ids.add(movie_id)

            movie_text = row.get("movie_text") or ""
            if movie_text.strip():
                text_keys.add(movie_text_key(movie_text))

        return ids, text_keys


def append_embeddings(output_csv: Path, rows: list[tuple[str, str, str]]) -> None:
    if not rows:
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()
    mode = "a" if file_exists else "w"

    with output_csv.open(mode, newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writerow(["movie_id", "movie_text", "embedding"])
        writer.writerows(rows)


def sanitize_text_for_embedding(text: str, max_chars: int) -> str:
    compact = " ".join((text or "").split())
    if max_chars <= 0 or len(compact) <= max_chars:
        return compact
    return compact[:max_chars]


def is_input_too_long_error(exc: BadRequestError) -> bool:
    return "maximum input length is 8192 tokens" in str(exc).lower()


def embed_single_with_retry(client: OpenAI, text: str, max_chars: int) -> tuple[str, list[float]]:
    candidate = sanitize_text_for_embedding(text, max_chars)

    for _ in range(6):
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=[candidate])
            return candidate, response.data[0].embedding
        except BadRequestError as exc:
            if not is_input_too_long_error(exc):
                raise

            if len(candidate) <= 512:
                raise

            candidate = candidate[: max(512, int(len(candidate) * 0.7))]

    raise RuntimeError("Failed to embed text after repeated truncation attempts.")


def embed_batch_with_fallback(
    client: OpenAI,
    batch: list[tuple[str, str]],
    max_chars: int,
) -> list[tuple[str, str, str]]:
    prepared_texts = [sanitize_text_for_embedding(movie_text, max_chars) for _, movie_text in batch]

    try:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=prepared_texts)
        return [
            (movie_id, prepared_text, build_vector_literal(item.embedding))
            for (movie_id, _), prepared_text, item in zip(batch, prepared_texts, response.data)
        ]
    except BadRequestError as exc:
        if not is_input_too_long_error(exc):
            raise

    rows_to_save: list[tuple[str, str, str]] = []
    for movie_id, movie_text in batch:
        prepared_text, embedding = embed_single_with_retry(client, movie_text, max_chars)
        rows_to_save.append((movie_id, prepared_text, build_vector_literal(embedding)))
    return rows_to_save


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")

    client = get_openai_client()
    movies = fetch_movies(args.input)

    if not movies:
        print(f"No movies found in {args.input}.")
        return

    print(f"Loaded {len(movies)} movies from {args.input}.")

    if args.overwrite and args.output.exists():
        args.output.unlink()

    precomputed_ids, precomputed_text_keys = load_precomputed_state(args.output)
    if precomputed_ids or precomputed_text_keys:
        print(
            f"Found {len(precomputed_ids)} precomputed ids and "
            f"{len(precomputed_text_keys)} precomputed text entries in {args.output}."
        )

    pending_movies: list[tuple[str, str]] = []
    seen_ids = set(precomputed_ids)
    seen_text_keys = set(precomputed_text_keys)
    for movie_id, movie_text in movies:
        text_key = movie_text_key(movie_text)
        if movie_id in seen_ids or text_key in seen_text_keys:
            continue
        pending_movies.append((movie_id, movie_text))
        seen_ids.add(movie_id)
        seen_text_keys.add(text_key)

    if args.limit > 0:
        pending_movies = pending_movies[: args.limit]

    if not pending_movies:
        print("No new movies to embed.")
        return

    print(f"Embedding {len(pending_movies)} pending movies.")
    print(f"Using batch size: {args.batch_size}.")

    total_written = 0
    for batch_number, batch in enumerate(chunked(pending_movies, args.batch_size), start=1):
        rows_to_save = embed_batch_with_fallback(client, batch, args.max_chars)
        append_embeddings(args.output, rows_to_save)
        total_written += len(rows_to_save)
        print(f"Embedded and saved batch {batch_number} with {len(rows_to_save)} rows.")

        # Small pause helps keep initial test runs gentle on the API.
        time.sleep(1.5)

    print(f"Wrote {total_written} new rows to {args.output}.")

    print("Embedding load complete.")


if __name__ == "__main__":
    main()
