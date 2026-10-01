"""
scripts/setup_data.py
~~~~~~~~~~~~~~~~~~~~~
Download the Spider dataset and prepare it for VERTO in one command.

    python scripts/setup_data.py
    python scripts/setup_data.py --output-dir /some/other/path
    python scripts/setup_data.py --force        # re-download everything
    python scripts/setup_data.py --verify-only  # just check completeness

Downloads the Spider dataset zip (~95 MB) from the community Google Drive
mirror used across text-to-SQL projects (no account required):
  - train_spider.json   7000 training examples
  - dev.json            1034 evaluation examples
  - database/           166 SQLite databases
  - tables.json         Spider's schema file (rebuilt from the SQLite schemas only if missing)
"""

from __future__ import annotations

# ── Path bootstrap (allows running without `pip install -e .`) ────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import shutil
import sqlite3


# ── tables.json reconstruction ────────────────────────────────────────────────

def _sqlite_type_to_spider(sqlite_type: str) -> str:
    t = sqlite_type.lower()
    if any(k in t for k in ("int", "num", "real", "float", "double", "dec", "bit")):
        return "number"
    if any(k in t for k in ("time", "date", "year")):
        return "time"
    if "bool" in t:
        return "others"
    return "text"


def build_tables_entry(db_id: str, db_path: Path) -> dict:
    """Introspect a SQLite database and return a Spider-format tables.json entry."""
    conn = sqlite3.connect(str(db_path))
    cur  = conn.cursor()

    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    table_names: list[str] = [row[0] for row in cur.fetchall()]

    column_names_original: list[list] = [[-1, "*"]]
    column_names: list[list]          = [[-1, "*"]]
    column_types: list[str]           = ["text"]
    primary_keys: list[int]           = []
    foreign_keys_raw: list[tuple]     = []
    col_global_idx = 1

    for table_idx, table_name in enumerate(table_names):
        cur.execute(f'PRAGMA table_info("{table_name}")')
        for cid, col_name, col_type, _, _, is_pk in cur.fetchall():
            column_names_original.append([table_idx, col_name])
            column_names.append([table_idx, col_name.lower()])
            column_types.append(_sqlite_type_to_spider(col_type or "text"))
            if is_pk:
                primary_keys.append(col_global_idx)
            col_global_idx += 1

        cur.execute(f'PRAGMA foreign_key_list("{table_name}")')
        for fk in cur.fetchall():
            _, _, ref_table, from_col, to_col, *_ = fk
            foreign_keys_raw.append((table_name, from_col, ref_table, to_col))

    conn.close()

    def find_col_idx(tname: str, cname: str) -> int | None:
        try:
            t_idx = table_names.index(tname)
        except ValueError:
            return None
        for g_idx, (t_i, c_n) in enumerate(column_names_original):
            if t_i == t_idx and c_n.lower() == cname.lower():
                return g_idx
        return None

    foreign_keys = []
    for from_t, from_c, to_t, to_c in foreign_keys_raw:
        fi = find_col_idx(from_t, from_c)
        ti = find_col_idx(to_t, to_c)
        if fi is not None and ti is not None:
            foreign_keys.append([fi, ti])

    return {
        "db_id":                 db_id,
        "table_names":           [t.lower() for t in table_names],
        "table_names_original":  table_names,
        "column_names":          column_names,
        "column_names_original": column_names_original,
        "column_types":          column_types,
        "primary_keys":          primary_keys,
        "foreign_keys":          foreign_keys,
    }


def build_tables_json(db_dir: Path, output_path: Path) -> None:
    print("\n[3/3] Building tables.json from SQLite schemas …")
    if output_path.exists():
        # Spider's own tables.json has hand-checked column types and foreign keys. One rebuilt
        # from the SQLite schemas differs on several databases, so it is only a fallback.
        print("  tables.json already exists — keeping it")
        return

    db_folders = sorted(
        f for f in db_dir.iterdir()
        if f.is_dir() and (f / f"{f.name}.sqlite").exists()
    )
    if not db_folders:
        print("  WARNING: No SQLite databases found — tables.json will be empty.")

    tables = []
    for i, db_folder in enumerate(db_folders, 1):
        db_id = db_folder.name
        print(f"  ({i:>3}/{len(db_folders)}) {db_id}")
        try:
            tables.append(build_tables_entry(db_id, db_folder / f"{db_id}.sqlite"))
        except Exception as exc:
            print(f"         WARNING — skipped ({exc})")

    with open(output_path, "w") as f:
        json.dump(tables, f, indent=2)
    print(f"  Saved tables.json  ({len(tables)} databases)")


# ── Download ──────────────────────────────────────────────────────────────────

def download(output_dir: Path, force: bool = False) -> bool:
    """Download, extract and index the dataset. Returns False if a step failed."""
    import urllib.request
    import urllib.parse
    import http.cookiejar
    import zipfile
    import re

    db_dir      = output_dir / "database"
    tables_json = output_dir / "tables.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "spider.zip"

    # Step 1: Download zip from Google Drive
    if not zip_path.exists() and not (tables_json.exists() and len(list(db_dir.iterdir())) > 1):
        file_id = "1TqleXec_OykOYFREKKtschzY29dUcVAQ"
        print(f"\n[1/3] Downloading Spider dataset zip from Google Drive (95 MB) …")
        
        cookie_jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
        urllib.request.install_opener(opener)
        
        url = f"https://docs.google.com/uc?export=download&id={file_id}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        try:
            with urllib.request.urlopen(req) as response:
                html = response.read().decode('utf-8', errors='ignore')
                
            confirm_token = None
            uuid_token = None
            
            confirm_match = re.search(r'name="confirm"\s+value="([^"]+)"', html)
            if confirm_match:
                confirm_token = confirm_match.group(1)
                
            uuid_match = re.search(r'name="uuid"\s+value="([^"]+)"', html)
            if uuid_match:
                uuid_token = uuid_match.group(1)
                
            if confirm_token and uuid_token:
                params = {
                    'id': file_id,
                    'export': 'download',
                    'confirm': confirm_token,
                    'uuid': uuid_token
                }
                download_url = f"https://drive.usercontent.google.com/download?{urllib.parse.urlencode(params)}"
            else:
                download_url = url
                
            req_download = urllib.request.Request(download_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req_download) as response:
                with open(zip_path, 'wb') as f:
                    bytes_downloaded = 0
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                        print(f"  Downloaded {bytes_downloaded / 1024 / 1024:.2f} MB...", end='\r')
                print("\n  Download complete.")
        except Exception as e:
            print(f"  ERROR: Download failed: {e}")
            if zip_path.exists():
                zip_path.unlink()
            return False
    else:
        print("\n[1/3] spider.zip or extracted database already exists — skipping download")

    # Step 2: Unzip and copy files
    print("\n[2/3] Extracting SQLite databases and schemas from ZIP …")
    try:
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as z:
                members = z.infolist()
                for member in members:
                    name = member.filename
                    if not name.startswith("spider/"):
                        continue
                    
                    # Strip the leading "spider/" directory
                    rel_path = name[len("spider/"):]
                    if not rel_path:
                        continue
                    
                    target_path = output_dir / rel_path
                    
                    if member.is_dir():
                        target_path.mkdir(parents=True, exist_ok=True)
                    else:
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        if not target_path.exists() or force:
                            with z.open(member) as source, open(target_path, "wb") as target:
                                shutil.copyfileobj(source, target)
            print("  Extraction complete.")
        else:
            print("  No zip file found to extract, assuming databases already present.")
    except Exception as e:
        print(f"  ERROR: Extraction failed: {e}")
        return False
    finally:
        # Cleanup ZIP file to save disk space
        if zip_path.exists():
            zip_path.unlink()

    # ── Step 3 ────────────────────────────────────────────────────────────────
    build_tables_json(db_dir, tables_json)
    return True



# ── Verification ──────────────────────────────────────────────────────────────

def verify(output_dir: Path) -> bool:
    required = [
        output_dir / "train_spider.json",
        output_dir / "dev.json",
        output_dir / "tables.json",
        output_dir / "database",
    ]
    print("\n── Verification ─────────────────────────────────────────")
    ok = True
    for path in required:
        status = "✓" if path.exists() else "✗"
        print(f"  {status}  {path}")
        if not path.exists():
            ok = False

    db_count = (
        sum(1 for d in (output_dir / "database").iterdir()
            if d.is_dir() and (d / f"{d.name}.sqlite").exists())
        if (output_dir / "database").exists() else 0
    )
    print(f"\n  SQLite databases found: {db_count}")
    print("─────────────────────────────────────────────────────────")

    if ok and db_count > 0:
        print("\n  ✓ Spider dataset ready. You can now run:\n")
        print("      python scripts/train.py")
        print("      verto --list-dbs\n")
    else:
        print("\n  ✗ Setup incomplete. Re-run: python scripts/setup_data.py\n")

    return ok and db_count > 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        prog="python scripts/setup_data.py",
        description="Download and prepare the Spider dataset for VERTO.",
    )
    p.add_argument("--output-dir",   default="data/spider", metavar="PATH",
                   help="Where to save the dataset  [default: data/spider]")
    p.add_argument("--force",        action="store_true",
                   help="Re-download and overwrite existing files with the dataset's own.")
    p.add_argument("--verify-only",  action="store_true",
                   help="Only check completeness; do not download.")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    print("VERTO — Spider Dataset Setup")
    print(f"Output directory: {output_dir.resolve()}")

    ok = True
    if not args.verify_only:
        ok = download(output_dir, force=args.force)

    ok = verify(output_dir) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
