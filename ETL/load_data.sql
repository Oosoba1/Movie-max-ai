
CREATE SCHEMA IF NOT EXISTS "public";
CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS "public"."genre_mapping" CASCADE;
DROP TABLE IF EXISTS "public"."movie_embeddings" CASCADE;
DROP TABLE IF EXISTS "public"."movies" CASCADE;
DROP TABLE IF EXISTS "public"."genre" CASCADE;

CREATE TABLE "public"."movies" (
    "Release Year" BIGINT,
    "Title" TEXT,
    "Origin/Ethnicity" TEXT,
    "Director" TEXT,
    "Cast" TEXT,
    "Genre" TEXT,
    "Wiki Page" TEXT,
    "Plot" TEXT,
    "id" BIGINT PRIMARY KEY
);

CREATE TABLE "public"."genre" (
    "genre_id" BIGINT PRIMARY KEY,
    "word" TEXT NOT NULL,
    "embedding" vector(1536),
    "pos" TEXT,
    "frequency" BIGINT NOT NULL
);

CREATE TABLE "public"."genre_mapping" (
    "line_no" BIGINT NOT NULL,
    "word_id" BIGINT NOT NULL,
    "word_position" BIGINT NOT NULL,
    "length" BIGINT NOT NULL,
    FOREIGN KEY ("line_no") REFERENCES "public"."movies" ("id"),
    FOREIGN KEY ("word_id") REFERENCES "public"."genre" ("genre_id")
);

CREATE TABLE "public"."movie_embeddings" (
    "movie_id" BIGINT PRIMARY KEY,
    "movie_text" TEXT NOT NULL,
    "embedding" vector(1536) NOT NULL,
    FOREIGN KEY ("movie_id") REFERENCES "public"."movies" ("id")
);

\! test -f data/movie_embedding.csv || printf '"movie_id","movie_text","embedding"\n' > data/movie_embedding.csv

\copy "public"."movies" ("Release Year", "Title", "Origin/Ethnicity", "Director", "Cast", "Genre", "Wiki Page", "Plot", "id") FROM 'data/movies.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')
\copy "public"."genre" ("genre_id", "word", "embedding", "pos", "frequency") FROM 'data/genre.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')
\copy "public"."genre_mapping" ("line_no", "word_id", "word_position", "length") FROM 'data/genre_mapping.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')
\copy "public"."movie_embeddings" ("movie_id", "movie_text", "embedding") FROM 'data/movie_embedding.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')