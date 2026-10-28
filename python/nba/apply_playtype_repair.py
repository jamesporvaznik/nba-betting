"""Run the reviewed play-type SQL using DB_PASSWORD or a private prompt.

Run locally with a hidden prompt, or configure DB_PASSWORD in python/.env.
The runner never prints the password or writes it to a file.
"""

from getpass import getpass
import json
from pathlib import Path
import sys

from dotenv import dotenv_values
import psycopg2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba.aggregate_playtypes import render_sql


def main():
    root = Path(__file__).resolve().parents[2]
    config = dotenv_values(root / "python" / ".env")
    settings = {
        "host": config.get("DB_HOST"),
        "port": config.get("DB_PORT") or "5432",
        "dbname": config.get("DB_NAME") or config.get("DB") or "postgres",
        "user": config.get("DB_USER"),
    }
    if not settings["host"] or not settings["user"]:
        print("Set DB_HOST and DB_USER in python/.env first.", file=sys.stderr)
        return 1
    if not config.get("DB_PASSWORD") and not sys.stdin.isatty():
        print("Run this yourself in an interactive terminal for the hidden password prompt.", file=sys.stderr)
        return 1

    repairs = root / "database" / "repairs"
    sql = (repairs / "player_playtypes_season_totals.sql").read_text()
    plan = json.loads((repairs / "player_playtypes_season_totals_review.json").read_text())
    # Ignore only the generated timestamp. Execute the entire original file.
    if sql.split("\n", 1)[1] != render_sql(plan).split("\n", 1)[1]:
        print("SQL does not match the reviewed plan; nothing was executed.", file=sys.stderr)
        return 1

    print("Applying the complete reviewed repair for 2022-23 through 2025-26.")
    print("All snapshot checks, updates, and result checks run in one transaction.")
    print("The runner does not save or print the password. No SQL is omitted or split.")
    password = config.get("DB_PASSWORD") or getpass("Supabase database password (hidden): ")
    if not password:
        print("No password entered; nothing was executed.", file=sys.stderr)
        return 1
    connection = None
    try:
        connection = psycopg2.connect(
            **settings, password=password, sslmode="require", connect_timeout=15,
            application_name="nba_playtype_repair",
        )
        password = None
        # The SQL file owns BEGIN/COMMIT and validates the result before COMMIT.
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute("SET statement_timeout = '120s'")
            cursor.execute(sql)
        print("Repair committed successfully; all in-transaction validations passed.")
        return 0
    except psycopg2.Error as exc:
        if connection is not None and not connection.closed:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("ROLLBACK")
            except psycopg2.Error:
                pass
        # Do not print connection strings or raw exceptions containing credentials.
        print(f"Database operation failed ({type(exc).__name__}).", file=sys.stderr)
        if exc.diag.message_primary:
            print(exc.diag.message_primary, file=sys.stderr)
        print("No successful commit was confirmed. Ask for a read-only verification before retrying.", file=sys.stderr)
        return 1
    finally:
        password = None
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    sys.exit(main())
