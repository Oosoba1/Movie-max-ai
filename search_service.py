import os

from openai import OpenAI

from config import SEARCH_RESULT_LIMIT
from db import get_db_connection


EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def build_vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(value) for value in embedding) + "]"


def get_page_number(request):
    """
    Read the current page number from the URL and keep it safe.

    This avoids invalid page values from causing messy pagination behavior.
    """
    page = request.args.get("page", "1").strip()

    if not page.isdigit():
        return 1

    return max(1, int(page))


def get_visible_page_numbers(current_page, total_pages, window_size=2):
    """
    Build a small list of page numbers around the current page.

    This keeps the pager compact instead of showing every page number.
    """
    start_page = max(1, current_page - window_size)
    end_page = min(total_pages, current_page + window_size)
    return list(range(start_page, end_page + 1))


def search_movies(query, current_page):

    movies = []
    error_message = None
    total_results = 0
    offset = (current_page - 1) * SEARCH_RESULT_LIMIT

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        if not query:
            cursor.execute("SELECT COUNT(*) AS total_results FROM movies")
            total_results = cursor.fetchone()["total_results"]

            cursor.execute(
                """
                SELECT
                    id,
                    "Title" AS title,
                    "Release Year" AS release_year,
                    "Origin/Ethnicity" AS origin,
                    "Director" AS director,
                    "Cast" AS cast_members,
                    "Genre" AS genre
                FROM movies
                ORDER BY "Release Year" DESC NULLS LAST, "Title" ASC
                LIMIT %s OFFSET %s
                """,
                (SEARCH_RESULT_LIMIT, offset),
            )
            movies = cursor.fetchall()
        else:
            cursor.execute("SELECT COUNT(*) AS total_results FROM movie_embeddings")
            total_results = cursor.fetchone()["total_results"]

            if total_results == 0:
                error_message = "No movie embeddings have been loaded yet."
            else:
                client = get_openai_client()
                response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
                query_vector = build_vector_literal(response.data[0].embedding)

                cursor.execute(
                    """
                    SELECT
                        o.id,
                        o."Title" AS title,
                        o."Release Year" AS release_year,
                        o."Origin/Ethnicity" AS origin,
                        o."Director" AS director,
                        o."Cast" AS cast_members,
                        o."Genre" AS genre
                    FROM movie_embeddings AS me
                    INNER JOIN movies AS o ON o.id = me.movie_id
                    ORDER BY me.embedding <=> %s::vector, o."Title" ASC
                    LIMIT %s OFFSET %s
                    """,
                    (query_vector, SEARCH_RESULT_LIMIT, offset),
                )
                movies = cursor.fetchall()
    except Exception as err:
        error_message = f"Database error: {err}"
    finally:
        if "cursor" in locals():
            cursor.close()
        if "connection" in locals():
            connection.close()

    total_pages = max(1, (total_results + SEARCH_RESULT_LIMIT - 1) // SEARCH_RESULT_LIMIT)

    return {
        "movies": movies,
        "error_message": error_message,
        "total_results": total_results,
        "total_pages": total_pages,
        "has_previous": current_page > 1,
        "has_next": current_page < total_pages,
        "visible_pages": get_visible_page_numbers(current_page, total_pages),
        "result_limit": SEARCH_RESULT_LIMIT,
    }
