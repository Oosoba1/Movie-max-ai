
import os

DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD"),
    "dbname": os.getenv("PGDATABASE", "movies_db"),
    "connect_timeout": 5,
}

SEARCH_RESULT_LIMIT = 100
