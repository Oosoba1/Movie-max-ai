
from db import get_db_connection

def get_movie_detail(movie_id):

    sql = """
        SELECT
            id,
            "Release Year" AS release_year,
            "Title" AS title,
            "Origin/Ethnicity" AS origin,
            "Director" AS director,
            "Cast" AS cast_members,
            "Genre" AS genre,
            "Wiki Page" AS wiki_page,
            "Plot" AS plot
        FROM movies
        WHERE id = %s
    """

    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, (movie_id,))
            return cursor.fetchone()
    finally:
        connection.close()
