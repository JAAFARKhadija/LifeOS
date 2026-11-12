
import sqlite3
from pathlib import Path

# Get the main project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Database location
DB_PATH = BASE_DIR / "data" / "lifeos.db"


def get_connection():
    """Create a connection to the SQLite database."""

    connection = sqlite3.connect(DB_PATH)

    # Enable foreign key constraints
    connection.execute("PRAGMA foreign_keys = ON")

    return connection
