"""Command-line runner for vacancy data collection.

Examples:
    # Collect 50 IT vacancies in Belarus
    python -m src.collection.runner --country BY --limit 50

    # Collect 100 vacancies across CIS for 'Python'
    python -m src.collection.runner --country RU --text Python --limit 100

    # Print total count of raw vacancies in database
    python -m src.collection.runner --stats
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.collection.collector import DEFAULT_IT_ROLES, VacancyCollector
from src.storage.raw_storage import RawStorage
from src.utils.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="JobPulse: Automated IT vacancy collector from HeadHunter API"
    )
    parser.add_argument(
        "--country",
        choices=["BY", "RU", "KZ"],
        default="BY",
        help="Target CIS country code (default: BY)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of new vacancies to collect (default: 50)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Optional search query filter (e.g. 'Python', 'DevOps')",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Display current statistics of collected vacancies and exit",
    )
    parser.add_argument(
        "--auth",
        action="store_true",
        help="Run interactive OAuth authorization flow with HH to obtain access token",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    storage = RawStorage(
        raw_data_dir=settings.raw_data_dir,
        sqlite_db_path=settings.sqlite_db_path,
    )

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.auth:
        from src.collection.auth import run_auth_flow
        run_auth_flow()
        sys.exit(0)

    if args.stats:
        total = storage.count_collected()
        print("\n--- JobPulse Staging Statistics ---")
        print(f"   Database path: {settings.sqlite_db_path}")
        print(f"   JSONL directory: {settings.raw_data_dir}")
        print(f"   Total raw vacancies stored: {total}\n")
        sys.exit(0)

    print(
        f"\n[START] Collecting JobPulse: Country={args.country}, Limit={args.limit}, Query='{args.text}'"
    )
    collector = VacancyCollector(storage=storage)

    stats = collector.collect(
        area_code=args.country,
        text=args.text,
        professional_roles=DEFAULT_IT_ROLES,
        max_vacancies=args.limit,
    )

    print("\n[DONE] Collection batch completed:")
    print(f"   Scanned in search: {stats['scanned']}")
    print(f"   Newly saved:       {stats['saved_new']}")
    print(f"   Skipped (exist):   {stats['skipped_existing']}")
    print(f"   Errors:            {stats['errors']}")
    print(f"   Total in database: {storage.count_collected()}\n")


if __name__ == "__main__":
    main()
