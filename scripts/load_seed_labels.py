"""Load the committed `fixtures/seed_labels.sql` file into MySQL.

Complements `scripts/generate_seed_labels.py`. The generator is the "rebuild
the fixture" path (re-runs the three smoke prompts, applies the perceptual
adjustment, and writes a fresh SQL file). This loader is the "use the
committed fixture" path: it simply runs the checked-in SQL against the
configured DB so anyone cloning the repo can reproduce the calibration run
deterministically.

Usage:
    python scripts/load_seed_labels.py               # load the default fixture
    python scripts/load_seed_labels.py --file other.sql
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.labels_store import _open_connection as open_labels_connection
from src.labels_store import ensure_labels_table

DEFAULT_FIXTURE_PATH = PROJECT_ROOT / "fixtures" / "seed_labels.sql"


def _split_statements(raw_sql: str) -> List[str]:
    """Split a multi-statement SQL file on ';' outside of quoted strings.

    Simpler than a full tokenizer but handles single/double quotes and
    doubled-up '' escape sequences, which is all our fixture emits.
    """
    statements: List[str] = []
    current_statement_chars: List[str] = []
    current_quote_char: str = ""
    index = 0
    raw_sql_length = len(raw_sql)
    while index < raw_sql_length:
        current_char = raw_sql[index]
        if current_quote_char:
            # Inside a quoted string.
            current_statement_chars.append(current_char)
            if current_char == current_quote_char:
                # Check for doubled '' which is an escape inside the string.
                if index + 1 < raw_sql_length and raw_sql[index + 1] == current_quote_char:
                    current_statement_chars.append(raw_sql[index + 1])
                    index += 2
                    continue
                current_quote_char = ""
            index += 1
            continue

        if current_char in ("'", '"'):
            current_quote_char = current_char
            current_statement_chars.append(current_char)
            index += 1
            continue
        if current_char == "-" and index + 1 < raw_sql_length and raw_sql[index + 1] == "-":
            # Line comment; skip to end of line.
            newline_index = raw_sql.find("\n", index)
            if newline_index == -1:
                break
            index = newline_index + 1
            continue
        if current_char == ";":
            statement_text = "".join(current_statement_chars).strip()
            if statement_text:
                statements.append(statement_text)
            current_statement_chars = []
            index += 1
            continue
        current_statement_chars.append(current_char)
        index += 1

    tail_statement = "".join(current_statement_chars).strip()
    if tail_statement:
        statements.append(tail_statement)
    return statements


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        default=str(DEFAULT_FIXTURE_PATH),
        help=f"Path to the SQL fixture (default: {DEFAULT_FIXTURE_PATH}).",
    )
    return parser.parse_args()


def main() -> int:
    cli_args = _parse_cli_args()
    fixture_path = Path(cli_args.file).resolve()
    if not fixture_path.exists():
        print(f"[seed-labels] fixture not found: {fixture_path}")
        return 1

    sql_text = fixture_path.read_text(encoding="utf-8")
    sql_statements = _split_statements(sql_text)
    if not sql_statements:
        print(f"[seed-labels] fixture is empty: {fixture_path}")
        return 1

    ensure_labels_table()

    executed_count = 0
    with open_labels_connection() as database_connection:
        with database_connection.cursor() as database_cursor:
            for single_statement in sql_statements:
                database_cursor.execute(single_statement)
                executed_count += 1

    print(
        f"[seed-labels] executed {executed_count} statement(s) from "
        f"{fixture_path.relative_to(PROJECT_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
