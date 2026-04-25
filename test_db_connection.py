"""Connect to MySQL using DB_HOST, DB_USER, DB_PASSWORD, DB_NAME from .env."""

import os
import sys
from pathlib import Path

import pymysql
from dotenv import load_dotenv

REQUIRED_ENV = ("DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME")
DEFAULT_MYSQL_PORT = 3306


def main() -> int:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")

    missing = [name for name in REQUIRED_ENV if not os.environ.get(name, "").strip()]
    if missing:
        print("Missing or empty in .env:", ", ".join(missing), file=sys.stderr)
        return 1

    port_raw = os.environ.get("DB_PORT", str(DEFAULT_MYSQL_PORT)).strip()
    try:
        port = int(port_raw)
    except ValueError:
        print(f"Invalid DB_PORT: {port_raw!r} (expected integer).", file=sys.stderr)
        return 1

    connect_kwargs = {
        "host": os.environ["DB_HOST"].strip(),
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
        "database": os.environ["DB_NAME"],
        "port": port,
        "connect_timeout": 10,
        "charset": "utf8mb4",
    }

    try:
        connection = pymysql.connect(**connect_kwargs)
    except pymysql.Error as exc:
        print("Connection failed:", exc, file=sys.stderr)
        return 1

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 AS ok")
            result_row = cursor.fetchone()
        assert result_row is not None, "no row from SELECT 1"
        assert int(result_row[0]) == 1, "unexpected SELECT 1 result"
    except (pymysql.Error, AssertionError, TypeError, ValueError) as exc:
        print("Query failed:", exc, file=sys.stderr)
        return 1
    finally:
        connection.close()

    print("MySQL connection OK. SELECT 1 =>", result_row[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
