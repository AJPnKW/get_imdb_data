#!/usr/bin/env python3
"""
merge_tmdb_json_to_csv.py

Purpose:
  1) Fix weird per-character slug directory/file names produced by an earlier slugger
     (e.g., "a_b_b_o_t_t_e_l_e_m_e_n_t_a_r_y_125935") into readable slugs
     (e.g., "abbott_elementary_125935").
  2) Merge all TMDB JSON files produced by tmdb_list_grabber into CSVs:
       - movies.csv
       - tv_shows.csv
       - tv_seasons.csv
       - tv_episodes.csv

Project-aware:
  - If this script is placed in Scripts/, project root is its parent.
  - Detects Results/ (or results/) and Logs/ (or logs/).
  - Chooses the latest run automatically unless --run-dir is provided.
  - Outputs CSVs to:
       <run_dir>/csv/
       <project_root>/Outputs/csv/    (created if missing)

Log file:
  Logs/run_merge_YYYYMMDD_HHMMSS.log

Usage examples:
  python merge_tmdb_json_to_csv.py
  python merge_tmdb_json_to_csv.py --run-dir "C:\path\to\Results\tmdb_download_YYYYMMDD_HHMMSS"
  python merge_tmdb_json_to_csv.py --fix-slugs-only
  python merge_tmdb_json_to_csv.py --no-fix

Author: ChatGPT (for Andrew)
"""

import os
import sys
import re
import csv
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# -----------------------------
# Path helpers & logging
# -----------------------------

def resolve_project_root(script_file: Path, override: Optional[str]) -> Path:
    if override:
        p = Path(override).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    sd = script_file.parent.resolve()
    if sd.name.lower() == "scripts":
        return sd.parent.resolve()
    return sd

def find_dir(root: Path, name_a: str, name_b: str) -> Optional[Path]:
    cand_a = root / name_a
    cand_b = root / name_b
    if cand_a.exists(): return cand_a
    if cand_b.exists(): return cand_b
    return None

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def setup_logging(logs_dir: Path) -> Path:
    ensure_dir(logs_dir)
    log_path = logs_dir / f"run_merge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"Logging to: {log_path}")
    return log_path

# -----------------------------
# Slug repair
# -----------------------------

def smart_slug(s: str) -> str:
    """
    Better slug: keep alnum, convert separators to single underscore, and collapse repeats.
    """
    out = []
    prev_us = False
    for ch in (s or "").lower().strip():
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    slug = "".join(out).strip("_")
    return slug or "unnamed"

def looks_like_per_char_slug(name: str) -> bool:
    """
    Detects patterns like 'a_b_b_o_t_t...'. This heuristic checks for underscores between
    many consecutive single letters or digits.
    """
    parts = name.split("_")
    # If there are many single-character segments (e.g., > 6) it's likely the buggy slug
    singles = sum(1 for p in parts if len(p) == 1)
    return len(parts) > 6 and singles / max(len(parts), 1) > 0.6

def fix_slug_name(name: str) -> str:
    """
    If name resembles per-character slug, recompose by removing underscores among singles,
    then re-slug normally with smart_slug.
    """
    if not looks_like_per_char_slug(name):
        return smart_slug(name)
    # collapse per-char segments until we hit a multi-char or numeric tail (like id)
    parts = name.split("_")
    buf = []
    current_word = []
    for p in parts:
        if len(p) == 1 and p.isalnum():
            current_word.append(p)
        else:
            if current_word:
                buf.append("".join(current_word))
                current_word = []
            buf.append(p)
    if current_word:
        buf.append("".join(current_word))
    recomposed = "_".join(buf)
    return smart_slug(recomposed)

def rename_buggy_paths(tv_root: Path) -> List[Tuple[Path, Path]]:
    """
    Walk tv_root and rename any directory/file that shows the buggy per-character slug pattern.
    Returns list of (old_path, new_path) renames performed.
    """
    renames = []

    # First pass: directories under tv_root
    for p in sorted(tv_root.glob("*")):
        if p.is_dir():
            new_name = fix_slug_name(p.name)
            if new_name != p.name:
                new_path = p.parent / new_name
                logging.info(f"Renaming dir: {p.name} -> {new_name}")
                if not new_path.exists():
                    p.rename(new_path)
                    renames.append((p, new_path))
                    p = new_path  # update reference
                else:
                    # collision: append suffix
                    suff = f"_{int(datetime.now().timestamp())}"
                    new_path = p.parent / (new_name + suff)
                    logging.info(f"Name collision; using: {new_path.name}")
                    p.rename(new_path)
                    renames.append((p, new_path))
                    p = new_path

            # Now rename files inside the (possibly renamed) dir
            for f in sorted(p.glob("*")):
                if f.is_file():
                    new_file = fix_slug_name(f.name)
                    if new_file != f.name:
                        target = f.parent / new_file
                        logging.info(f"Renaming file: {f.name} -> {new_file}")
                        if not target.exists():
                            f.rename(target)
                            renames.append((f, target))
                        else:
                            suff = f"_{int(datetime.now().timestamp())}"
                            target = f.parent / (Path(new_file).stem + suff + f.suffix)
                            f.rename(target)
                            renames.append((f, target))

    return renames

# -----------------------------
# CSV merge
# -----------------------------

def load_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logging.warning(f"Failed to load JSON {p}: {e}")
        return None

def dict_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def join_names(items: List[Dict[str, Any]], key="name", sep="|") -> str:
    vals = []
    for it in items or []:
        v = it.get(key)
        if v:
            vals.append(str(v))
    return sep.join(vals)

def find_latest_run(results_dir: Path) -> Optional[Path]:
    if not results_dir.exists():
        return None
    runs = [p for p in results_dir.iterdir() if p.is_dir() and p.name.lower().startswith("tmdb_download_")]
    if not runs:
        return None
    runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return runs[0]

def movies_to_csv(movies_dir: Path, out_csv: Path) -> int:
    rows = []
    for jf in sorted(movies_dir.glob("*.json")):
        data = load_json(jf)
        if not data: 
            continue
        rows.append({
            "id": data.get("id"),
            "title": data.get("title") or data.get("original_title"),
            "original_title": data.get("original_title"),
            "release_date": data.get("release_date"),
            "runtime": data.get("runtime"),
            "status": data.get("status"),
            "genres": join_names(data.get("genres")),
            "spoken_languages": join_names(data.get("spoken_languages")),
            "production_companies": join_names(data.get("production_companies")),
            "vote_average": data.get("vote_average"),
            "vote_count": data.get("vote_count"),
            "popularity": data.get("popularity"),
            "imdb_id": data.get("imdb_id"),
        })
    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [
            "id","title","original_title","release_date","runtime","status",
            "genres","spoken_languages","production_companies","vote_average","vote_count","popularity","imdb_id"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logging.info(f"Wrote movies CSV: {out_csv} ({len(rows)} rows)")
    return len(rows)

def tv_shows_to_csv(tv_dir: Path, out_csv: Path) -> int:
    rows = []
    for show_dir in sorted(tv_dir.glob("*")):
        if not show_dir.is_dir():
            continue
        show_meta = next(show_dir.glob("show_*.json"), None)
        if not show_meta:
            continue
        data = load_json(show_meta)
        if not data:
            continue
        rows.append({
            "id": data.get("id"),
            "name": data.get("name") or data.get("original_name"),
            "original_name": data.get("original_name"),
            "first_air_date": data.get("first_air_date"),
            "last_air_date": data.get("last_air_date"),
            "number_of_seasons": data.get("number_of_seasons"),
            "number_of_episodes": data.get("number_of_episodes"),
            "status": data.get("status"),
            "in_production": data.get("in_production"),
            "genres": join_names(data.get("genres")),
            "networks": join_names(data.get("networks")),
            "origin_country": "|".join(data.get("origin_country") or []),
            "languages": "|".join(data.get("languages") or []),
            "vote_average": data.get("vote_average"),
            "vote_count": data.get("vote_count"),
            "popularity": data.get("popularity"),
        })
    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [
            "id","name","original_name","first_air_date","last_air_date","number_of_seasons","number_of_episodes",
            "status","in_production","genres","networks","origin_country","languages","vote_average","vote_count","popularity"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logging.info(f"Wrote tv_shows CSV: {out_csv} ({len(rows)} rows)")
    return len(rows)

def tv_seasons_to_csv(tv_dir: Path, out_csv: Path) -> int:
    rows = []
    for show_dir in sorted(tv_dir.glob("*")):
        if not show_dir.is_dir():
            continue
        # load show meta for name/id
        show_meta = next(show_dir.glob("show_*.json"), None)
        base = load_json(show_meta) if show_meta else {}
        show_id = base.get("id")
        show_name = base.get("name") or base.get("original_name")

        seasons_idx = show_dir / "seasons_index.json"
        s_data = load_json(seasons_idx) or []
        for s in s_data:
            rows.append({
                "show_id": show_id,
                "show_name": show_name,
                "season_number": s.get("season_number"),
                "name": s.get("name"),
                "air_date": s.get("air_date"),
                "episode_count": s.get("episode_count"),
            })
    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [
            "show_id","show_name","season_number","name","air_date","episode_count"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logging.info(f"Wrote tv_seasons CSV: {out_csv} ({len(rows)} rows)")
    return len(rows)

def tv_episodes_to_csv(tv_dir: Path, out_csv: Path) -> int:
    rows = []
    for show_dir in sorted(tv_dir.glob("*")):
        if not show_dir.is_dir():
            continue
        # show meta (for id & name)
        show_meta = next(show_dir.glob("show_*.json"), None)
        base = load_json(show_meta) if show_meta else {}
        show_id = base.get("id")
        show_name = base.get("name") or base.get("original_name")

        for season_file in sorted(show_dir.glob("season_*.json")):
            sdata = load_json(season_file) or {}
            snum = sdata.get("season_number")
            for ep in sdata.get("episodes") or []:
                rows.append({
                    "show_id": show_id,
                    "show_name": show_name,
                    "season_number": snum,
                    "episode_number": ep.get("episode_number"),
                    "name": ep.get("name"),
                    "air_date": ep.get("air_date"),
                    "runtime": ep.get("runtime"),
                    "vote_average": ep.get("vote_average"),
                    "vote_count": ep.get("vote_count"),
                })
    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [
            "show_id","show_name","season_number","episode_number","name",
            "air_date","runtime","vote_average","vote_count"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logging.info(f"Wrote tv_episodes CSV: {out_csv} ({len(rows)} rows)")
    return len(rows)

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Fix slugs and merge TMDB JSONs to CSVs.")
    ap.add_argument("--project-root", help="Project root (defaults to parent of Scripts/ or script folder).")
    ap.add_argument("--run-dir", help="Specific tmdb_download_... folder. If omitted, pick the latest under Results/.")
    ap.add_argument("--no-fix", action="store_true", help="Skip slug repair step.")
    ap.add_argument("--fix-slugs-only", action="store_true", help="Only fix slugs, do not generate CSVs.")
    args = ap.parse_args()

    script_file = Path(__file__).resolve()
    project_root = resolve_project_root(script_file, args.project_root)
    logging_dir = find_dir(project_root, "Logs", "logs") or ensure_dir(project_root / "Logs")
    results_dir = find_dir(project_root, "Results", "results") or ensure_dir(project_root / "Results")
    outputs_dir = ensure_dir(project_root / "Outputs" / "csv")

    log_path = setup_logging(logging_dir)
    logging.info(f"Project root: {project_root}")

    # Determine run directory
    run_dir = Path(args.run_dir).resolve() if args.run_dir else (find_latest_run(results_dir) or results_dir)
    if not run_dir.exists():
        logging.error(f"Run directory not found: {run_dir}")
        sys.exit(2)

    logging.info(f"Using run directory: {run_dir}")

    # Directories inside run
    movies_dir = run_dir / "movies"
    tv_dir = run_dir / "tv"
    if not movies_dir.exists() and not tv_dir.exists():
        logging.error("No 'movies' or 'tv' directories found in the run directory. Nothing to process.")
        sys.exit(2)

    # 1) Fix buggy slugs (if not disabled)
    if not args.no_fix:
        if tv_dir.exists():
            renames = rename_buggy_paths(tv_dir)
            if renames:
                logging.info(f"Renamed {len(renames)} paths to fix slugs.")
            else:
                logging.info("No slug fixes were necessary.")
        else:
            logging.info("No tv directory found; skipping slug fixes.")

    if args.fix_slugs_only:
        logging.info("Slug-fix-only requested. Exiting.")
        return

    # 2) Merge to CSVs
    csv_run_dir = ensure_dir(run_dir / "csv")

    # Movies
    if movies_dir.exists():
        movies_cnt = movies_to_csv(movies_dir, csv_run_dir / "movies.csv")
        # also copy to Outputs/csv
        try:
            (outputs_dir / "movies.csv").write_text((csv_run_dir / "movies.csv").read_text(encoding="utf-8"), encoding="utf-8")
        except Exception as e:
            logging.warning(f"Failed to copy movies.csv to Outputs/csv: {e}")
    else:
        logging.info("No movies directory; skipping movies.csv")

    # TV shows
    if tv_dir.exists():
        shows_cnt = tv_shows_to_csv(tv_dir, csv_run_dir / "tv_shows.csv")
        seasons_cnt = tv_seasons_to_csv(tv_dir, csv_run_dir / "tv_seasons.csv")
        episodes_cnt = tv_episodes_to_csv(tv_dir, csv_run_dir / "tv_episodes.csv")
        # copy to Outputs/csv
        for name in ("tv_shows.csv","tv_seasons.csv","tv_episodes.csv"):
            try:
                (outputs_dir / name).write_text((csv_run_dir / name).read_text(encoding="utf-8"), encoding="utf-8")
            except Exception as e:
                logging.warning(f"Failed to copy {name} to Outputs/csv: {e}")
    else:
        logging.info("No tv directory; skipping tv CSVs.")

    logging.info("All done.")

    print("\n=== SUMMARY ===")
    print(f"Project root : {project_root}")
    print(f"Run dir      : {run_dir}")
    print(f"Logs         : {log_path}")
    if (run_dir / 'csv').exists():
        print(f"CSV (run)    : {run_dir / 'csv'}")
    if outputs_dir.exists():
        print(f"CSV (Outputs): {outputs_dir}")

if __name__ == "__main__":
    main()
