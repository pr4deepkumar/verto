"""
verto/cli.py
~~~~~~~~~~~~
CLI entrypoint registered as the ``verto`` console script via pyproject.toml.

Supports dual execution modes:
1. **Remote Cloud Mode**: Connects over HTTP to a cloud-hosted VERTO Dashboard.
   Does not require local PyTorch, GPU, model checkpoints, or Spider dataset files.
   Triggered via `--server-url <url>` or setting `VERTO_SERVER_URL=<url>`.
2. **Local Engine Mode**: Loads PyTorch & NLDIEngine directly for offline inference.
   Used with `--local` (never falls back to a server), or when no server URL is set.

Examples:
    verto --server-url http://cloud-host:8000 --list-dbs
    verto --server-url http://cloud-host:8000 --db concert_singer "How many singers are older than 30?"
    verto --health
    verto --local --db concert_singer "Find all stadiums"
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

from verto import config as cfg
from verto.api.client import VERTOClient


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_result(result: dict, json_output: bool = False) -> None:
    if json_output:
        print(json.dumps(result, indent=2, default=str))
        return

    sql      = result.get("sql") or "(none)"
    rows     = result.get("result")
    columns  = result.get("columns")
    attempts = result.get("attempts", 0)
    error    = result.get("error")

    print()
    print(f"  SQL        : {sql}")
    print(f"  Attempts   : {attempts}")

    if error:
        print(f"  Error      : {error}", file=sys.stderr)
    elif rows is not None:
        if columns:
            print(f"  Columns    : {', '.join(columns)}")
        if rows:
            print(f"  Rows       : {len(rows)}")
            for i, row in enumerate(rows[:20]):
                print(f"    [{i:>3}] {row}")
            if len(rows) > 20:
                print(f"    … and {len(rows) - 20} more rows")
        else:
            print("  Result     : (empty — query returned no rows)")
    print()


def _local_databases(spider_dir: str) -> list[str]:
    """Database IDs from tables.json, without loading the model."""
    from verto.engine.schema import load_tables_raw

    return sorted(db["db_id"] for db in load_tables_raw(Path(spider_dir) / "tables.json"))


def _print_local_health(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    print("\n  Local Engine Mode:")
    print(f"    Checkpoint Path    : {checkpoint} ({'found' if checkpoint.exists() else 'missing'})")
    print(f"    Spider Directory   : {args.spider_dir} ({len(_local_databases(args.spider_dir))} databases)")
    print(f"    Device             : {args.device or 'auto'}\n")


def _load_engine(args: argparse.Namespace):
    try:
        from verto.engine.inference import NLDIEngine

        return NLDIEngine(checkpoint_path=args.checkpoint, spider_dir=args.spider_dir, device=args.device)
    except ImportError as e:
        print(f"\n  [Error] Local mode needs the model packages: pip install -e \".[local]\" ({e})\n",
              file=sys.stderr)
        sys.exit(1)


# ── Parser ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verto",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            VERTO — Natural Language Database Interface Engine
            ==================================================
            Translate plain-English questions into SQL and execute them
            against Spider SQLite databases. Supports local engine or Cloud API.

            Examples
            --------
              # Query Cloud Dashboard
              verto --server-url http://<cloud-ip>:8000 --list-dbs
              verto -s http://<cloud-ip>:8000 --db concert_singer "How many singers are older than 30?"

              # Query Local Engine
              verto --local --db concert_singer "How many singers are older than 30?"
              verto --health
            """
        ),
    )

    parser.add_argument("question", nargs="?", metavar="QUESTION",
                        help="Plain-English question to translate into SQL.")
    parser.add_argument("--db", metavar="DB_ID",
                        help="Spider database ID to query.")
    parser.add_argument("--list-dbs", action="store_true",
                        help="Print all available database IDs and exit.")
    parser.add_argument("--health", action="store_true",
                        help="Check health status of the Cloud server or local environment.")
    parser.add_argument("-s", "--server-url", metavar="URL", default=cfg.SERVER_URL,
                        help="Cloud Dashboard API URL (e.g. http://<cloud-ip>:8000 or VERTO_SERVER_URL)")
    parser.add_argument("--local", action="store_true",
                        help="Force local PyTorch engine execution instead of remote Cloud server.")
    parser.add_argument("--no-schema-linking", action="store_true",
                        help="Disable schema-linking (pass the full schema to the model).")
    parser.add_argument("--no-recovery", action="store_true",
                        help="Run only the model's top query; don't fall back to its other candidates.")
    parser.add_argument("--json", dest="json_output", action="store_true",
                        help="Output result as JSON (useful for scripting).")
    parser.add_argument("--checkpoint", default=cfg.CHECKPOINT_PATH, metavar="PATH",
                        help=f"Path to best_model.pt [default: {cfg.CHECKPOINT_PATH}]")
    parser.add_argument("--spider-dir", default=cfg.SPIDER_DIR, metavar="PATH",
                        help=f"Spider dataset root [default: {cfg.SPIDER_DIR}]")
    parser.add_argument("--device", default=cfg.DEVICE, metavar="DEVICE",
                        help='Torch device, e.g. "cuda", "mps" or "cpu" [default: auto]')
    return parser


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # Check the arguments before anything slow: loading the model takes a while.
    if not (args.health or args.list_dbs or args.question):
        parser.print_help()
        sys.exit(1)
    if args.question and not (args.health or args.list_dbs) and not args.db:
        parser.error("--db DB_ID is required when a QUESTION is given.")

    remote = bool(args.server_url) and not args.local
    if not remote and not (Path(args.spider_dir) / "tables.json").exists():
        if args.local:
            print(
                f"\n  [Error] --local needs the Spider data, but {args.spider_dir} has no tables.json.\n"
                "  Run setup locally:  python scripts/setup_data.py\n",
                file=sys.stderr,
            )
        else:
            print(
                "\n  [Error] No local Spider dataset found and no Cloud server URL provided.\n"
                "  Options:\n"
                "    1. Query a Cloud Dashboard:  verto -s http://<cloud-ip>:8000 --db <db> \"<question>\"\n"
                "    2. Set environment variable:  export VERTO_SERVER_URL=http://<cloud-ip>:8000\n"
                "    3. Run setup locally:         python scripts/setup_data.py\n",
                file=sys.stderr,
            )
        sys.exit(1)

    if args.health:
        if remote:
            try:
                info = VERTOClient(args.server_url).health()
                print("\n  Cloud Server Health:")
                print(f"    Status             : {info.get('status')}")
                print(f"    Device             : {info.get('device')}")
                print(f"    Checkpoint Loaded  : {info.get('checkpoint_loaded')}")
                print(f"    Databases Available: {info.get('database_count')}\n")
            except Exception as e:
                print(f"\n  [Error] Cloud server health check failed: {e}\n", file=sys.stderr)
                sys.exit(1)
        else:
            _print_local_health(args)
        sys.exit(0)

    if args.list_dbs:
        try:
            dbs = VERTOClient(args.server_url).list_databases() if remote else _local_databases(args.spider_dir)
            target_info = f"cloud server {args.server_url}" if remote else "local dataset"
            print(f"\n{len(dbs)} databases available from {target_info}:\n")
            for db in dbs:
                print(f"  {db}")
            print()
            sys.exit(0)
        except Exception as e:
            print(f"\n  [Error] Failed to list databases: {e}\n", file=sys.stderr)
            sys.exit(1)

    if not args.json_output:
        print(f"\n  Question   : {args.question}")
        print(f"  Database   : {args.db}")
        if remote:
            print(f"  Target     : Cloud Server ({args.server_url})")

    runner = VERTOClient(args.server_url) if remote else _load_engine(args)
    result = runner.query(
        question=args.question,
        db_id=args.db,
        use_schema_linking=not args.no_schema_linking,
        use_error_recovery=not args.no_recovery,
    )

    _print_result(result, json_output=args.json_output)

    if result.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
