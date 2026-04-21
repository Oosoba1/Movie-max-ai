# MovieMax

MovieMax is a Flask movie discovery app backed by PostgreSQL. The current app is being simplified around one main experience: a user types a natural-language movie request and the backend returns the closest movie matches.

## Project structure

- `app.py`
  - Flask routes for home, search, and movie detail pages
- `config.py`
  - shared configuration values
- `db.py`
  - shared PostgreSQL connection helper
- `catalog_service.py`
  - single-movie detail reads
- `search_service.py`
  - search and pagination logic
- `ETL/movie_embedding_loader.py`
  - builds `data/movie_embedding.csv` from `data/movies.csv`
- `templates/`
  - HTML templates for the homepage, results page, and movie detail page
- `static/`
  - frontend styling

Data Setup
The movie dataset used in this project comes from Kaggle and was loaded into PostgreSQL for use in the application.

## Requirements

- Python 3.9+
- PostgreSQL
- `pgvector` enabled in the target database

## Environment setup

Create a local `.env` or export variables in your terminal.

Example values:

```bash
export PGHOST='localhost'
export PGUSER='postgres'
export PGPASSWORD='your_postgres_password'
export PGDATABASE='movies_db'
export OPENAI_API_KEY='your_openai_api_key'
export OPENAI_EMBEDDING_MODEL='text-embedding-3-small'
export EMBED_BATCH_SIZE=1
export EMBED_MAX_MOVIES=1
```

You can also copy `.env.example` to `.env` and fill in your own values.

## Install dependencies

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run the app

```bash
.venv/bin/python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Load embeddings

This script reads movies from `data/movies.csv`, sends movie text to OpenAI embeddings, and writes vectors to `data/movie_embedding.csv`.

```bash
uv run python -m ETL.movie_embedding_loader
```

Optional flags:

- `--input /path/to/movies.csv`
- `--output /path/to/movie_embedding.csv`
- `--limit 100` (use `0` or a negative value for all rows)
- `--batch-size 50` for larger or smaller chunking per API request
- `--max-chars 24000` to trim long text before embedding
- `--overwrite` to rebuild output instead of resuming

The loader is resumable: if `movie_embedding.csv` already exists, it skips precomputed `movie_id` values and appends new rows after each batch.

Start with a very small batch size and movie limit to control cost.

## ETL

Run the ETL workflow without genre embeddings:

```bash
uv run python -m ETL.run_etl --input /path/to/raw/movies.csv
```

Run the ETL workflow and also populate the `embedding` column in `data/genre.csv`:

```bash
uv run python -m ETL.run_etl --input /path/to/raw/movies.csv --with-embeddings
```

Load generated CSV data into PostgreSQL on demand:

```bash
uv run python -m ETL.load_data
```

By default, this command initializes the schema from `ETL/schema.sql` before loading CSV data.

If you prefer SQL, use `ETL/load_data.sql` (defaults to `data/*.csv`) and override paths when needed:

```bash
psql "$DATABASE_URL" \
  -v movies_csv='/absolute/path/to/movies.csv' \
  -v genre_csv='/absolute/path/to/genre.csv' \
  -v genre_mapping_csv='/absolute/path/to/genre_mapping.csv' \
  -f ETL/load_data.sql
```

Optional flags:

- `--movies /path/to/movies.csv`
- `--genre /path/to/genre.csv`
- `--mapping /path/to/genre_mapping.csv`
- `--schema /path/to/schema.sql`
- `--skip-schema-init` to skip running schema SQL
- `--skip-truncate` to append instead of truncating destination tables first
