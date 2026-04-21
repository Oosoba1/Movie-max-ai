import argparse
import csv
import io
import os
import re
import tarfile
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

try:
    import nltk
    from nltk.corpus import wordnet
except ModuleNotFoundError:
    nltk = None
    wordnet = None

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = Any


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ARCHIVE_PATH = DATA_DIR / "data.tgz"
NLTK_DATA_DIR = DATA_DIR / "nltk_data"
GENRE_FIELDNAMES = ["genre_id", "word", "embedding", "pos", "frequency"]
CSV_WRITE_KWARGS = {
    "doublequote": True,
    "lineterminator": "\n",
    "quotechar": '"',
    "quoting": csv.QUOTE_ALL,
}

NON_WORD_RE = re.compile(r"[^a-z0-9\s]+")
SOURCE_SCI_FI_RE = re.compile(r"\bscience[\s-]*fiction\b")
THREE_D_RE = re.compile(r"\b3[\s-]*d\b")
WHITESPACE_RE = re.compile(r"\s+")
MIN_PARTIAL_WORD_LENGTH = 3

SCI_FI_RE = re.compile(r"\bsci\s*-\s*fi\b")
SCIENCE_FICTION_RE = re.compile(r"\bscience\s*-\s*fiction\b")
NON_ALPHANUMERIC_RE = re.compile(r"[^a-z0-9]+")

WORDNET_POS_MAP = {
    "n": "NOUN",
    "v": "VERB",
    "a": "ADJ",
    "s": "ADJ",
    "r": "ADV",
}
POS_ORDER = {
    "NOUN": 0,
    "VERB": 1,
    "ADJ": 2,
    "ADV": 3,
    "X": 4,
}

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "10"))


def input_csv_path(value: str) -> Path:
    return Path(value).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ETL workflow end to end: import movies, build genre words, "
            "annotate part of speech, and optionally generate embeddings."
        )
    )
    parser.add_argument(
        "--input",
        type=input_csv_path,
        help="Optional raw source CSV path. Defaults to ./movies.csv or data/data.tgz.",
    )
    parser.add_argument("--movies-out", type=Path, default=DATA_DIR / "movies.csv")
    parser.add_argument("--genre-out", type=Path, default=DATA_DIR / "genre.csv")
    parser.add_argument("--mapping-out", type=Path, default=DATA_DIR / "genre_mapping.csv")
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Also populate the embedding column in genre.csv.",
    )
    return parser.parse_args()


def normalize_genre_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "genre_id": (row.get("genre_id") or row.get("id") or "").strip(),
        "word": (row.get("word") or "").strip(),
        "embedding": (row.get("embedding") or "").strip(),
        "pos": (row.get("pos") or row.get("possible_parts_of_speech") or "").strip(),
        "frequency": (row.get("frequency") or "").strip(),
    }


def build_genre_row(
    genre_id: int | str,
    word: str,
    *,
    embedding: str = "",
    pos: str = "",
    frequency: int | str = "",
) -> dict[str, str]:
    return normalize_genre_row(
        {
            "genre_id": str(genre_id),
            "word": word,
            "embedding": embedding,
            "pos": pos,
            "frequency": str(frequency),
        }
    )


def read_genre_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError("The input CSV is missing a header row.")
        if "word" not in reader.fieldnames or ("genre_id" not in reader.fieldnames and "id" not in reader.fieldnames):
            raise ValueError("The input CSV must contain genre_id/id and word columns.")

        return [normalize_genre_row(row) for row in reader]


@contextmanager
def open_movies_source(
    input_path: Path | None,
    *,
    include_generated_output: bool = True,
    exclude_paths: Iterable[Path] = (),
) -> Iterator[TextIO]:
    excluded_paths = {path.resolve(strict=False) for path in exclude_paths}
    candidates = []
    if input_path is not None:
        candidates.append(input_path)
    else:
        candidates.append(ROOT_DIR / "movies.csv")
        if include_generated_output:
            candidates.append(DATA_DIR / "movies.csv")

    for candidate in candidates:
        if candidate.resolve(strict=False) in excluded_paths:
            continue

        if candidate.exists():
            with candidate.open("r", newline="", encoding="utf-8") as fh:
                yield fh
            return

    if ARCHIVE_PATH.resolve(strict=False) not in excluded_paths and ARCHIVE_PATH.exists():
        with tarfile.open(ARCHIVE_PATH, "r:gz") as archive:
            member = archive.extractfile("movies.csv")
            if member is None:
                raise FileNotFoundError(f"movies.csv was not found inside {ARCHIVE_PATH}.")
            with io.TextIOWrapper(member, encoding="utf-8", newline="") as fh:
                yield fh
            return

    searched = [str(path) for path in candidates] + [str(ARCHIVE_PATH)]
    raise FileNotFoundError(f"Could not find an input CSV. Checked: {', '.join(searched)}")


def normalize_source_genre_text(value: str) -> str:
    normalized = (value or "").strip().lower()
    normalized = SOURCE_SCI_FI_RE.sub("scifi", normalized)
    normalized = THREE_D_RE.sub("3d", normalized)
    normalized = NON_WORD_RE.sub(" ", normalized)
    return WHITESPACE_RE.sub(" ", normalized).strip()


def extract_word_parts(word: str) -> list[str]:
    if not word:
        return []

    if len(word) <= MIN_PARTIAL_WORD_LENGTH:
        return [word]

    return [word[:end_index] for end_index in range(MIN_PARTIAL_WORD_LENGTH, len(word) + 1)]


def extract_source_genre_word_parts(value: str) -> list[tuple[int, str]]:
    normalized = normalize_source_genre_text(value)
    if not normalized:
        return []

    word_parts: list[tuple[int, str]] = []
    seen: set[str] = set()

    for word_order, word in enumerate(normalized.split(), start=1):
        for part in extract_word_parts(word):
            if part in seen:
                continue

            word_parts.append((word_order, part))
            seen.add(part)

    return word_parts


def extract_source_genre_words(value: str) -> list[str]:
    return [word for _, word in extract_source_genre_word_parts(value)]


def escape_output_value(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\n", r"\n")


def build_source_outputs(
    input_path: Path | None,
    movies_out: Path,
    genre_out: Path,
    mapping_out: Path,
) -> tuple[int, int, int]:
    if input_path is not None and input_path.resolve(strict=False) == movies_out.resolve(strict=False):
        raise ValueError(
            "The source CSV and movies output resolve to the same file. "
            "Pass a raw source CSV via --input or write the output to a different path."
        )

    for path in [movies_out, genre_out, mapping_out]:
        path.parent.mkdir(parents=True, exist_ok=True)

    movies_tmp = movies_out.with_name(f"{movies_out.name}.tmp")
    genre_tmp = genre_out.with_name(f"{genre_out.name}.tmp")
    mapping_tmp = mapping_out.with_name(f"{mapping_out.name}.tmp")
    tmp_paths = [movies_tmp, genre_tmp, mapping_tmp]

    movie_count = 0
    mapping_count = 0
    genre_ids: dict[str, int] = {}

    try:
        try:
            with open_movies_source(
                input_path,
                include_generated_output=input_path is not None,
                exclude_paths=[movies_out],
            ) as source_fh:
                reader = csv.DictReader(source_fh)
                if not reader.fieldnames:
                    raise ValueError("The input CSV is missing a header row.")
                if "Genre" not in reader.fieldnames:
                    raise ValueError("The input CSV does not contain a Genre column.")

                movie_fieldnames = [field for field in reader.fieldnames if field != "line_no"]
                if "id" not in movie_fieldnames:
                    movie_fieldnames.append("id")

                with movies_tmp.open("w", newline="", encoding="utf-8") as movies_fh, mapping_tmp.open(
                    "w", newline="", encoding="utf-8"
                ) as mapping_fh:
                    movies_writer = csv.DictWriter(movies_fh, fieldnames=movie_fieldnames, **CSV_WRITE_KWARGS)
                    mapping_writer = csv.writer(mapping_fh, **CSV_WRITE_KWARGS)

                    movies_writer.writeheader()
                    mapping_writer.writerow(["line_no", "word", "word_order"])

                    for row_number, row in enumerate(reader, start=1):
                        line_no = (row.get("line_no") or "").strip() or str(row_number)
                        row["id"] = line_no

                        movies_writer.writerow(
                            {field: escape_output_value(row.get(field, "")) for field in movie_fieldnames}
                        )
                        movie_count += 1

                        for word_order, word in extract_source_genre_word_parts(row.get("Genre", "")):
                            if word not in genre_ids:
                                genre_ids[word] = len(genre_ids) + 1
                            mapping_writer.writerow(
                                [escape_output_value(line_no), escape_output_value(word), word_order]
                            )
                            mapping_count += 1
        except FileNotFoundError as err:
            if input_path is None:
                raise FileNotFoundError(
                    "Could not find a raw source movie CSV. "
                    f"The generator will not reuse its own output file at {movies_out}. "
                    f"Pass --input /path/to/raw/movies.csv or restore {ARCHIVE_PATH}."
                ) from err
            raise

        with genre_tmp.open("w", newline="", encoding="utf-8") as genre_fh:
            genre_writer = csv.DictWriter(genre_fh, fieldnames=GENRE_FIELDNAMES, **CSV_WRITE_KWARGS)
            genre_writer.writeheader()
            for word, genre_id in genre_ids.items():
                genre_writer.writerow(
                    build_genre_row(
                        genre_id=genre_id,
                        word=escape_output_value(word),
                    )
                )

        movies_tmp.replace(movies_out)
        mapping_tmp.replace(mapping_out)
        genre_tmp.replace(genre_out)
    except Exception:
        for path in tmp_paths:
            path.unlink(missing_ok=True)
        raise

    return movie_count, mapping_count, len(genre_ids)


def normalize_genre_text(value: str) -> str:
    normalized = (value or "").strip().lower()
    normalized = SCIENCE_FICTION_RE.sub("scifi", normalized)
    normalized = SCI_FI_RE.sub("scifi", normalized)
    return NON_ALPHANUMERIC_RE.sub(" ", normalized)


def extract_genre_words(value: str) -> list[str]:
    return normalize_genre_text(value).split()


def write_movies_with_line_no(input_path: Path, movies_out: Path) -> int:
    if input_path.resolve(strict=False) == movies_out.resolve(strict=False):
        raise ValueError("The source CSV and movies output resolve to the same file.")

    movies_out.parent.mkdir(parents=True, exist_ok=True)
    movies_tmp = movies_out.with_name(f"{movies_out.name}.tmp")
    row_count = 0

    try:
        with input_path.open("r", newline="", encoding="utf-8") as source_fh:
            reader = csv.DictReader(source_fh)
            if not reader.fieldnames:
                raise ValueError("The input CSV is missing a header row.")

            movie_fieldnames = [field for field in reader.fieldnames if field != "line_no"]
            movie_fieldnames.append("line_no")

            with movies_tmp.open("w", newline="", encoding="utf-8") as movies_fh:
                writer = csv.DictWriter(movies_fh, fieldnames=movie_fieldnames, **CSV_WRITE_KWARGS)
                writer.writeheader()

                for line_no, row in enumerate(reader, start=1):
                    output_row = {field: row.get(field, "") for field in movie_fieldnames if field != "line_no"}
                    output_row["line_no"] = str(line_no)
                    writer.writerow(output_row)
                    row_count += 1

        movies_tmp.replace(movies_out)
    except Exception:
        movies_tmp.unlink(missing_ok=True)
        raise

    return row_count


def write_movies_csv(input_path: Path | None, output_path: Path) -> int:
    if input_path is not None and input_path.resolve(strict=False) == output_path.resolve(strict=False):
        raise ValueError("The source CSV and movies output resolve to the same file.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output_path.with_name(f"{output_path.name}.tmp")
    row_count = 0

    try:
        with open_movies_source(
            input_path,
            include_generated_output=input_path is not None,
            exclude_paths=[output_path],
        ) as source_fh:
            reader = csv.DictReader(source_fh)
            if not reader.fieldnames:
                raise ValueError("The input CSV is missing a header row.")

            movie_fieldnames = [field for field in reader.fieldnames if field not in {"line_no", "id"}]
            movie_fieldnames.append("id")

            with output_tmp.open("w", newline="", encoding="utf-8") as output_fh:
                writer = csv.DictWriter(output_fh, fieldnames=movie_fieldnames, **CSV_WRITE_KWARGS)
                writer.writeheader()

                for row_number, row in enumerate(reader, start=1):
                    movie_id = (row.get("id") or row.get("line_no") or "").strip() or str(row_number)
                    output_row = {
                        field: escape_output_value(row.get(field, ""))
                        for field in movie_fieldnames
                        if field != "id"
                    }
                    output_row["id"] = escape_output_value(movie_id)
                    writer.writerow(output_row)
                    row_count += 1

        output_tmp.replace(output_path)
    except Exception:
        output_tmp.unlink(missing_ok=True)
        raise

    return row_count


def build_genre_outputs(
    input_path: Path,
    words_out: Path,
    mapping_out: Path,
) -> tuple[int, int, int]:
    if input_path.resolve(strict=False) == words_out.resolve(strict=False):
        raise ValueError("The source CSV and words output resolve to the same file.")

    if input_path.resolve(strict=False) == mapping_out.resolve(strict=False):
        raise ValueError("The source CSV and mapping output resolve to the same file.")

    words_out.parent.mkdir(parents=True, exist_ok=True)
    mapping_out.parent.mkdir(parents=True, exist_ok=True)

    words_tmp = words_out.with_name(f"{words_out.name}.tmp")
    mapping_tmp = mapping_out.with_name(f"{mapping_out.name}.tmp")
    tmp_paths = [words_tmp, mapping_tmp]

    line_count = 0
    mapping_count = 0
    word_ids: dict[str, int] = {}
    word_frequencies: Counter[str] = Counter()

    try:
        with input_path.open("r", newline="", encoding="utf-8") as source_fh:
            reader = csv.DictReader(source_fh)
            if not reader.fieldnames:
                raise ValueError("The input CSV is missing a header row.")
            if "Genre" not in reader.fieldnames:
                raise ValueError("The input CSV does not contain a Genre column.")

            with mapping_tmp.open("w", newline="", encoding="utf-8") as mapping_fh:
                mapping_writer = csv.writer(mapping_fh, **CSV_WRITE_KWARGS)
                mapping_writer.writerow(["line_no", "word_id", "word_position", "length"])

                for line_no, row in enumerate(reader, start=1):
                    line_count += 1
                    words = extract_genre_words(row.get("Genre", ""))
                    length = len(words)

                    for word_position, word in enumerate(words, start=1):
                        if word not in word_ids:
                            word_ids[word] = len(word_ids) + 1
                        word_frequencies[word] += 1

                        mapping_writer.writerow([line_no, word_ids[word], word_position, length])
                        mapping_count += 1

        with words_tmp.open("w", newline="", encoding="utf-8") as words_fh:
            words_writer = csv.DictWriter(words_fh, fieldnames=GENRE_FIELDNAMES, **CSV_WRITE_KWARGS)
            words_writer.writeheader()
            for word, genre_id in word_ids.items():
                words_writer.writerow(
                    build_genre_row(
                        genre_id=genre_id,
                        word=word,
                        frequency=word_frequencies[word],
                    )
                )

        words_tmp.replace(words_out)
        mapping_tmp.replace(mapping_out)
    except Exception:
        for path in tmp_paths:
            path.unlink(missing_ok=True)
        raise

    return line_count, mapping_count, len(word_ids)


def has_nltk_resource(resource_path: str) -> bool:
    for candidate in [resource_path, f"{resource_path}.zip"]:
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            continue

    return False


def ensure_nltk_resources() -> None:
    if nltk is None or wordnet is None:
        raise RuntimeError("nltk is not installed.")

    nltk_data_path = str(NLTK_DATA_DIR)
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_path)

    missing_packages: list[str] = []
    for resource_path, package_name in [
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]:
        if not has_nltk_resource(resource_path):
            missing_packages.append(package_name)

    if not missing_packages:
        return

    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for package_name in missing_packages:
        download_succeeded = nltk.download(package_name, download_dir=str(NLTK_DATA_DIR), quiet=True)
        if not download_succeeded:
            raise RuntimeError(
                "Failed to download required NLTK data. "
                "Run `.venv/bin/python -c \"import nltk; "
                f"nltk.download('{package_name}', download_dir='data/nltk_data')\"`."
            )


def normalize_possible_parts_of_speech(parts_of_speech: set[str]) -> list[str]:
    cleaned = {part for part in parts_of_speech if part}
    if not cleaned:
        return ["X"]

    return sorted(cleaned, key=lambda part: (POS_ORDER.get(part, len(POS_ORDER)), part))


def lookup_wordnet_parts_of_speech(word: str) -> set[str]:
    if not word:
        return set()

    ensure_nltk_resources()
    return {
        WORDNET_POS_MAP[synset.pos()]
        for synset in wordnet.synsets(word)
        if synset.pos() in WORDNET_POS_MAP
    }


def normalize_tagged_part_of_speech(tag: str) -> str | None:
    if tag.startswith("NN"):
        return "NOUN"
    if tag.startswith("VB"):
        return "VERB"
    if tag.startswith("JJ"):
        return "ADJ"
    if tag.startswith("RB"):
        return "ADV"

    return None


def guess_tagged_part_of_speech(word: str) -> str | None:
    if not word:
        return None

    ensure_nltk_resources()
    tag = nltk.pos_tag([word])[0][1]
    return normalize_tagged_part_of_speech(tag)


def infer_parts_of_speech_for_sequence(words: list[str]) -> list[str | None]:
    if not words:
        return []

    ensure_nltk_resources()
    return [normalize_tagged_part_of_speech(tag) for _, tag in nltk.pos_tag(words)]


def order_contextual_parts_of_speech(part_counts: Counter[str]) -> list[str]:
    return sorted(
        part_counts,
        key=lambda part: (-part_counts[part], POS_ORDER.get(part, len(POS_ORDER)), part),
    )


def resolve_mapping_word(
    row: dict[str, str],
    genre_words_by_id: dict[str, str],
    *,
    id_field: str | None,
) -> str:
    if "word" in row:
        return (row.get("word") or "").strip().lower()
    if not id_field:
        return ""
    return genre_words_by_id.get((row.get(id_field) or "").strip(), "")


def infer_contextual_parts_of_speech_by_word(
    mapping_path: Path | None,
    genre_words_by_id: dict[str, str],
) -> dict[str, list[str]]:
    if mapping_path is None or not mapping_path.exists():
        return {}

    with mapping_path.open("r", newline="", encoding="utf-8") as mapping_fh:
        reader = csv.DictReader(mapping_fh)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError("The mapping CSV is missing a header row.")

        order_field = (
            "word_position"
            if "word_position" in fieldnames
            else "word_order" if "word_order" in fieldnames else None
        )
        id_field = (
            "word_id"
            if "word_id" in fieldnames
            else "genre_id" if "genre_id" in fieldnames else None
        )

        if "line_no" not in fieldnames or order_field is None or ("word" not in fieldnames and id_field is None):
            raise ValueError(
                "The mapping CSV must contain line_no, an order column, and either a word or word identifier column."
            )

        sequences_by_line_no: dict[str, list[tuple[int, str]]] = defaultdict(list)

        for row in reader:
            line_no = (row.get("line_no") or "").strip()
            raw_order = (row.get(order_field) or "").strip()
            word = resolve_mapping_word(row, genre_words_by_id, id_field=id_field)

            if not line_no or not raw_order or not word:
                continue

            try:
                word_order = int(raw_order)
            except ValueError:
                continue

            sequences_by_line_no[line_no].append((word_order, word))

    all_context_counts: dict[str, Counter[str]] = defaultdict(Counter)
    multiword_context_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for sequence in sequences_by_line_no.values():
        words = [word for _, word in sorted(sequence, key=lambda item: item[0])]
        contextual_parts = infer_parts_of_speech_for_sequence(words)

        for word, part_of_speech in zip(words, contextual_parts):
            if not part_of_speech:
                continue

            all_context_counts[word][part_of_speech] += 1
            if len(words) > 1:
                multiword_context_counts[word][part_of_speech] += 1

    contextual_parts_by_word: dict[str, list[str]] = {}
    for word in set(all_context_counts) | set(multiword_context_counts):
        preferred_counts = multiword_context_counts[word] or all_context_counts[word]
        contextual_parts_by_word[word] = order_contextual_parts_of_speech(preferred_counts)

    return contextual_parts_by_word


def possible_parts_of_speech_for_word(
    word: str,
    contextual_parts_of_speech: list[str] | None = None,
) -> list[str]:
    normalized_word = (word or "").strip().lower()
    if contextual_parts_of_speech:
        return contextual_parts_of_speech

    possible_parts_of_speech = lookup_wordnet_parts_of_speech(normalized_word)

    if not possible_parts_of_speech:
        guessed_part_of_speech = guess_tagged_part_of_speech(normalized_word)
        if guessed_part_of_speech:
            possible_parts_of_speech.add(guessed_part_of_speech)

    return normalize_possible_parts_of_speech(possible_parts_of_speech)


def annotate_genre_outputs(input_path: Path, output_path: Path, mapping_path: Path | None = None) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output_path.with_name(f"{output_path.name}.tmp")
    row_count = 0

    try:
        rows = read_genre_rows(input_path)
        genre_words_by_id = {
            row["genre_id"]: row["word"].lower()
            for row in rows
            if row["genre_id"] and row["word"]
        }
        contextual_parts_by_word = infer_contextual_parts_of_speech_by_word(mapping_path, genre_words_by_id)

        with output_tmp.open("w", newline="", encoding="utf-8") as output_fh:
            writer = csv.DictWriter(output_fh, fieldnames=GENRE_FIELDNAMES, **CSV_WRITE_KWARGS)
            writer.writeheader()

            for row in rows:
                normalized_word = row["word"].lower()
                parts_of_speech = possible_parts_of_speech_for_word(
                    row["word"],
                    contextual_parts_by_word.get(normalized_word),
                )
                updated_row = dict(row)
                updated_row["pos"] = "|".join(parts_of_speech)
                writer.writerow(updated_row)
                row_count += 1

        output_tmp.replace(output_path)
    except Exception:
        output_tmp.unlink(missing_ok=True)
        raise

    return row_count


def get_openai_client() -> OpenAI:
    if OpenAI is Any:
        raise RuntimeError("openai is not installed.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def build_genre_rows_from_movies(input_path: Path | None) -> list[dict[str, str]]:
    genre_ids: dict[str, int] = {}
    word_frequencies: Counter[str] = Counter()

    with open_movies_source(input_path) as source_fh:
        reader = csv.DictReader(source_fh)
        if not reader.fieldnames:
            raise ValueError("The input CSV is missing a header row.")
        if "Genre" not in reader.fieldnames:
            raise ValueError("The input CSV does not contain a Genre column.")

        for row in reader:
            for word in extract_source_genre_words(row.get("Genre", "")):
                if word not in genre_ids:
                    genre_ids[word] = len(genre_ids) + 1
                word_frequencies[word] += 1

    return [
        build_genre_row(
            genre_id=genre_id,
            word=word,
            frequency=word_frequencies[word],
        )
        for word, genre_id in genre_ids.items()
    ]


def write_genre_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = path.with_name(f"{path.name}.tmp")

    try:
        with output_tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=GENRE_FIELDNAMES, **CSV_WRITE_KWARGS)
            writer.writeheader()
            writer.writerows(rows)

        output_tmp.replace(path)
    except Exception:
        output_tmp.unlink(missing_ok=True)
        raise


def build_vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(value) for value in embedding) + "]"


def chunked(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def populate_embeddings(rows: list[dict[str, str]], client: OpenAI) -> list[dict[str, str]]:
    if not rows:
        return rows

    rows_by_id = {row["genre_id"]: dict(row) for row in rows}

    for batch_number, batch in enumerate(chunked(rows, BATCH_SIZE), start=1):
        texts = [row["word"] for row in batch]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)

        for row, item in zip(batch, response.data):
            rows_by_id[row["genre_id"]]["embedding"] = build_vector_literal(item.embedding)

        print(f"Embedded batch {batch_number} ({len(batch)} genre words).")
        time.sleep(0.2)

    return [rows_by_id[row["genre_id"]] for row in rows]


def run_pipeline(
    *,
    input_path: Path | None,
    movies_out: Path,
    genre_out: Path,
    mapping_out: Path,
    with_embeddings: bool = False,
) -> None:
    movie_count = write_movies_csv(input_path=input_path, output_path=movies_out)
    print(f"Wrote {movie_count} movies to {movies_out}.")

    line_count, mapping_count, genre_count = build_genre_outputs(
        input_path=movies_out,
        words_out=genre_out,
        mapping_out=mapping_out,
    )
    print(f"Processed {line_count} input rows from {movies_out}.")
    print(f"Wrote {genre_count} unique words to {genre_out}.")
    print(f"Wrote {mapping_count} word mappings to {mapping_out}.")

    pos_count = annotate_genre_outputs(
        input_path=genre_out,
        output_path=genre_out,
        mapping_path=mapping_out,
    )
    print(f"Wrote pos for {pos_count} genre words to {genre_out}.")

    if not with_embeddings:
        return

    rows = read_genre_rows(genre_out)
    if not rows:
        print(f"No genre words found for {genre_out}.")
        return

    client = get_openai_client()
    updated_rows = populate_embeddings(rows, client)
    write_genre_rows(genre_out, updated_rows)
    print(f"Wrote embeddings for {len(updated_rows)} genre words to {genre_out}.")


def main() -> None:
    args = parse_args()
    run_pipeline(
        input_path=args.input,
        movies_out=args.movies_out,
        genre_out=args.genre_out,
        mapping_out=args.mapping_out,
        with_embeddings=args.with_embeddings,
    )


if __name__ == "__main__":
    main()
