#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
tmdb_list_grabber.py
Author: ChatGPT (for Andrew)

What it does
------------
• Fetch TMDB v4 lists (movies + TV).
• TV: save show JSON + seasons_index.json + seasons JSON for last & future/undated seasons.
• Clean & rename TV folders (fix per-character underscore slugs; preserve .json; recover missing .json).
• Merge JSON → CSV (movies, tv_shows, tv_seasons, tv_episodes) into run\csv and Outputs\csv.
• Build a single "combined_all_rows.csv" (movies + shows + seasons + episodes) with row_type.
• GUI (PyQt5) with background threads (no freezing), progress bar, completion dialog, clickable links.
• Quiet console (only major steps); detailed logs in Logs\*.log (use --verbose for chatty console).

New in this version
-------------------
• Adds image URL columns:
    - movies/shows: poster_url, backdrop_url
    - seasons: season_poster_url
    - episodes: episode_still_url
• Adds external IDs (IMDb, TVDB, etc.) where available.
• Adds "where to watch" provider columns for your REGION (default CA):
    - providers_flatrate_CA, providers_rent_CA, providers_buy_CA, providers_ads_CA
• Keeps all earlier fixes (single dialog, proper progress to 100%, better status text).

Note about "Watched" status
---------------------------
Your personal watched/favorite/watchlist/rated state is only available via TMDB "Account States" endpoints
and requires a user session_id (TMDB login flow). The API key and read token alone are not enough.
If you want, we can add a simple "Sign in to TMDB" in the GUI later to fetch & include those columns.

Install deps:
    python -m pip install requests PyQt5
"""

import os
import sys
import csv
import json
import time
import traceback
import logging
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
from pathlib import Path

import requests

# GUI
try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False

# ========= CONFIG =========
# (pre-populated as requested; can still be overridden by env vars or CLI)
PASTE_TMDB_READ_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI3YTBiNTY4OTgxMjEyYjhjNGM1NzE0NzNkYzg3ZTE3YSIsIm5iZiI6MTczNTg3NTI3Ny4yNDg5OTk4LCJzdWIiOiI2Nzc3NWFjZDgyY2NlMTVhNzY3NDk1NjEiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.sk5RdizDtejRX6cdRkv6cj__X8z6nLbXMuq9kJ0h-7s"
PASTE_TMDB_API_KEY    = "7a0b568981212b8c4c571473dc87e17a"

DEFAULT_LIST_IDS      = [8511658, 8511657]  # Andrew's lists
LANG   = "en-CA"
REGION = "CA"  # used for where-to-watch provider columns

TMDB_V4_BASE = "https://api.themoviedb.org/4"
TMDB_V3_BASE = "https://api.themoviedb.org/3"
REQUEST_TIMEOUT = 20
RETRY_COUNT     = 3
RETRY_BACKOFF_S = 2

# TMDB image base (see https://developer.themoviedb.org/docs/image-basics)
IMG_BASE = "https://image.tmdb.org/t/p/"
IMG_POSTER_SIZE   = "w500"
IMG_BACKDROP_SIZE = "w780"
IMG_STILL_SIZE    = "w300"
IMG_SEASON_POSTER_SIZE = "w342"

# Modern palette (Shopify-ish)
THEME = {
    "bg":"#FAF7F2","panel":"#FFFFFF","text":"#0C2A1E","subtext":"#4D6B5C",
    "primary":"#064E3B","primary2":"#0E7A5F","accent":"#10B981","border":"#D7E2DB"
}

CURRENT_LOG_PATH: Optional[Path] = None

# ======= Logging =======
def setup_logging(log_dir: Path, prefix: str = "run", verbose_console: bool = False) -> Path:
    """Quiet console (WARNING) by default, full DEBUG to file."""
    global CURRENT_LOG_PATH
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if verbose_console else logging.WARNING)
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)

    progress_logger = logging.getLogger("progress")
    progress_logger.setLevel(logging.INFO)
    progress_logger.propagate = True

    CURRENT_LOG_PATH = log_path
    logging.getLogger(__name__).debug("Logging initialized.")
    return log_path

def progress(msg: str):
    logging.getLogger("progress").info(msg)
    logging.getLogger(__name__).info(msg)

def debug(msg: str):
    logging.getLogger(__name__).debug(msg)

# ======= Path helpers =======
def find_or_create_dir(root: Path, preferred: str, alt: str) -> Path:
    """Prefer existing name; else create preferred."""
    cand1 = root / preferred
    cand2 = root / alt
    if cand1.exists(): return cand1
    if cand2.exists(): return cand2
    cand1.mkdir(parents=True, exist_ok=True)
    return cand1

def resolve_project_root(script_file: Path, override: Optional[str]) -> Path:
    if override:
        p = Path(override).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    sd = script_file.parent.resolve()
    return sd.parent.resolve() if sd.name.lower() == "scripts" else sd

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

# ======= Slug repair (keep .json, recover if missing) =======
def smart_slug(s: str) -> str:
    out = []; prev_us = False
    for ch in (s or "").lower().strip():
        if ch.isalnum():
            out.append(ch); prev_us = False
        else:
            if not prev_us:
                out.append("_"); prev_us = True
    return "".join(out).strip("_") or "unnamed"

def looks_like_per_char_slug(name: str) -> bool:
    parts = name.split("_")
    singles = sum(1 for p in parts if len(p) == 1)
    return len(parts) > 6 and singles / max(len(parts), 1) > 0.6

def fix_slug_name_stem(stem: str) -> str:
    """Return a cleaned stem (no extension). If it was a_b_b_o_t_t, collapse to abbott, etc."""
    if not looks_like_per_char_slug(stem):
        return smart_slug(stem)
    parts = stem.split("_")
    buf, cur = [], []
    for p in parts:
        if len(p) == 1 and p.isalnum():
            cur.append(p)
        else:
            if cur:
                buf.append("".join(cur)); cur = []
            buf.append(p)
    if cur: buf.append("".join(cur))
    return smart_slug("_".join(buf))

def rename_buggy_paths(tv_root: Path) -> List[Tuple[Path, Path]]:
    """
    Repair per-character slugs but preserve file extensions.
    Returns a list of (old_path, new_path).
    """
    renames: List[Tuple[Path, Path]] = []
    if not tv_root.exists(): return renames

    for p in sorted(tv_root.glob("*")):
        if p.is_dir():
            # Fix directory name
            new_dir_name = fix_slug_name_stem(p.name)
            if new_dir_name != p.name:
                target = p.parent / new_dir_name
                if target.exists():
                    target = p.parent / f"{new_dir_name}_{int(datetime.now().timestamp())}"
                p.rename(target)
                renames.append((p, target))
                p = target

            # Fix files inside (keep .json)
            for f in sorted(p.glob("*")):
                if f.is_file():
                    stem, suffix = f.stem, f.suffix
                    new_stem = fix_slug_name_stem(stem)
                    if new_stem != stem:
                        target = f.with_name(new_stem + suffix)
                        if target.exists():
                            target = f.with_name(f"{new_stem}_{int(datetime.now().timestamp())}{suffix}")
                        f.rename(target)
                        renames.append((f, target))
    return renames

def recover_missing_json_extensions(tv_root: Path) -> int:
    """
    If older versions stripped '.json' (e.g., 'seasons_index_json'), restore '.json'
    for expected TMDB files.
    """
    fixed = 0
    if not tv_root.exists(): return fixed
    for f in tv_root.rglob("*"):
        if not f.is_file() or f.suffix.lower() == ".json":
            continue
        name = f.name.lower()
        # Patterns that should end with .json
        if name.startswith("show_") or name.startswith("season_") or "seasons_index" in name:
            target = f.with_name(f.stem + ".json")
            if target.exists():
                target = f.with_name(f"{f.stem}_{int(datetime.now().timestamp())}.json")
            f.rename(target)
            fixed += 1
    return fixed

# ======= TMDB client =======
@dataclass
class TMDBClient:
    read_token: str
    api_key_v3: Optional[str] = None

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.read_token}", "Content-Type": "application/json;charset=utf-8"}

    def _params(self) -> Dict[str, str]:
        p: Dict[str, str] = {}
        if self.api_key_v3:
            p["api_key"] = self.api_key_v3
        if LANG:
            p["language"] = LANG
        return p

    def _request(self, method: str, url: str, params: Dict[str, Any] = None) -> Any:
        params = params or {}
        last_exc = None
        for attempt in range(1, RETRY_COUNT + 1):
            try:
                debug(f"REQ {method} {url} params={params}")
                resp = requests.request(
                    method, url, headers=self._headers(),
                    params={**self._params(), **params},
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.status_code >= 400:
                    snippet = (resp.text or "")[:500]
                    logging.warning(f"HTTP {resp.status_code} for {url} -> {snippet}")
                    if 500 <= resp.status_code < 600:
                        raise requests.RequestException(f"Server error {resp.status_code}")
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_exc = e
                logging.warning(f"Attempt {attempt}/{RETRY_COUNT} failed for {url}: {e}")
                if attempt < RETRY_COUNT:
                    time.sleep(RETRY_BACKOFF_S * attempt)
        raise last_exc

    # API methods (details)
    def list_items(self, list_id: int, page: int = 1) -> Dict[str, Any]:
        return self._request("GET", f"{TMDB_V4_BASE}/list/{list_id}", params={"page": page})

    def movie_details(self, movie_id: int) -> Dict[str, Any]:
        # Use append_to_response for richer payload where it’s stable
        return self._request("GET", f"{TMDB_V3_BASE}/movie/{movie_id}",
                             params={"append_to_response": "release_dates,credits,images,external_ids"})

    def tv_details(self, tv_id: int) -> Dict[str, Any]:
        return self._request("GET", f"{TMDB_V3_BASE}/tv/{tv_id}",
                             params={"append_to_response": "credits,content_ratings,images,external_ids"})

    def tv_season(self, tv_id: int, season_number: int) -> Dict[str, Any]:
        # Season detail includes episodes and their still_path
        return self._request("GET", f"{TMDB_V3_BASE}/tv/{tv_id}/season/{season_number}", params={})

    # API methods (providers & external ids when needed separately)
    def movie_watch_providers(self, movie_id: int) -> Dict[str, Any]:
        return self._request("GET", f"{TMDB_V3_BASE}/movie/{movie_id}/watch/providers", params={})

    def tv_watch_providers(self, tv_id: int) -> Dict[str, Any]:
        return self._request("GET", f"{TMDB_V3_BASE}/tv/{tv_id}/watch/providers", params={})

# ======= Fetch logic =======
def parse_date(s: Optional[str]) -> Optional[date]:
    if not s: return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y"):
        try: return datetime.strptime(s, fmt).date()
        except Exception: pass
    try: return datetime.fromisoformat(s).date()
    except Exception: return None

def list_all_items(client: TMDBClient, list_id: int) -> List[Dict[str, Any]]:
    debug(f"Fetching list {list_id}...")
    first = client.list_items(list_id, page=1)
    total_pages = max(1, int(first.get("total_pages") or 1))
    results = list(first.get("results") or [])
    debug(f"List {list_id}: page 1/{total_pages}, items={len(results)}")
    for p in range(2, total_pages + 1):
        page_data = client.list_items(list_id, page=p)
        page_results = page_data.get("results") or []
        results.extend(page_results)
        debug(f"List {list_id}: page {p}, +{len(page_results)} (total {len(results)})")
    return results

def url_or_blank(base: str, size: str, path: Optional[str]) -> str:
    return f"{base}{size}{path}" if path else ""

def extract_providers_for_region(prov_json: Dict[str, Any], region: str) -> Dict[str, str]:
    # prov_json format: {"results": {"CA": {"link": "...", "flatrate":[...], "rent":[...], "buy":[...], "ads":[...]}}}
    r = (prov_json or {}).get("results", {}).get(region, {})
    def join_names(key: str) -> str:
        return "|".join(p.get("provider_name") for p in r.get(key, []) if p.get("provider_name"))
    return {
        f"providers_flatrate_{region}": join_names("flatrate"),
        f"providers_rent_{region}":     join_names("rent"),
        f"providers_buy_{region}":      join_names("buy"),
        f"providers_ads_{region}":      join_names("ads"),
        f"providers_link_{region}":     r.get("link") or ""
    }

def save_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    debug(f"Wrote JSON: {path}")

def process_movie(client: TMDBClient, item: Dict[str, Any], movies_dir: Path) -> Dict[str, Any]:
    movie_id = int(item.get("id"))
    details = client.movie_details(movie_id)
    # attach providers for REGION (separate endpoint for stability)
    providers = client.movie_watch_providers(movie_id)

    title = details.get("title") or details.get("original_title") or f"movie_{movie_id}"
    out = movies_dir / f"{smart_slug(title)}_{movie_id}.json"

    # Enrich detail JSON with computed URLs and providers summary for convenience
    enriched = dict(details)
    enriched["_poster_url"]   = url_or_blank(IMG_BASE, IMG_POSTER_SIZE,   details.get("poster_path"))
    enriched["_backdrop_url"] = url_or_blank(IMG_BASE, IMG_BACKDROP_SIZE, details.get("backdrop_path"))
    enriched["_providers"]    = extract_providers_for_region(providers, REGION)

    save_json(out, enriched)
    return {"type": "movie", "id": movie_id, "title": title, "path": str(out.resolve())}

def process_tv(client: TMDBClient, item: Dict[str, Any], tv_root: Path) -> Dict[str, Any]:
    tv_id = int(item.get("id"))
    details = client.tv_details(tv_id)
    providers = client.tv_watch_providers(tv_id)

    name = details.get("name") or details.get("original_name") or f"tv_{tv_id}"
    show_dir = tv_root / f"{smart_slug(name)}_{tv_id}"
    ensure_dir(show_dir)

    enriched = dict(details)
    enriched["_poster_url"]   = url_or_blank(IMG_BASE, IMG_POSTER_SIZE,   details.get("poster_path"))
    enriched["_backdrop_url"] = url_or_blank(IMG_BASE, IMG_BACKDROP_SIZE, details.get("backdrop_path"))
    enriched["_providers"]    = extract_providers_for_region(providers, REGION)

    save_json(show_dir / f"show_{smart_slug(name)}_{tv_id}.json", enriched)

    seasons = details.get("seasons") or []
    # also record a poster URL for each season entry if present
    seasons_enriched = []
    for s in seasons:
        s2 = dict(s)
        s2["_season_poster_url"] = url_or_blank(IMG_BASE, IMG_SEASON_POSTER_SIZE, s.get("poster_path"))
        seasons_enriched.append(s2)
    save_json(show_dir / "seasons_index.json", seasons_enriched)

    nums = [s.get("season_number") for s in seasons if isinstance(s.get("season_number"), int)]
    last_num = max(nums) if nums else None

    today = date.today()
    future_nums = set()
    for s in seasons:
        sn = s.get("season_number")
        ad = parse_date(s.get("air_date"))
        if isinstance(sn, int) and (ad is None or ad > today):
            future_nums.add(sn)

    targets = set()
    if last_num is not None:
        targets.add(last_num)
    targets |= future_nums

    saved = []
    for sn in sorted(targets):
        try:
            sdata = client.tv_season(tv_id, sn)
            # enrich each episode with still URL
            for ep in (sdata.get("episodes") or []):
                ep["_still_url"] = url_or_blank(IMG_BASE, IMG_STILL_SIZE, ep.get("still_path"))
            out = show_dir / f"season_{sn:02d}.json"
            save_json(out, sdata)
            saved.append(sn)
        except Exception as e:
            logging.warning(f"Season fetch failed tv={tv_id} s{sn}: {e}")

    return {
        "type": "tv",
        "id": tv_id,
        "name": name,
        "path": str((show_dir / f"show_{smart_slug(name)}_{tv_id}.json").resolve()),
        "seasons_saved": saved,
        "last_season_number": last_num,
        "future_or_undated": sorted(future_nums),
    }

# ======= CSV helpers =======
def load_json_safe(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logging.warning(f"Failed to load JSON {p}: {e}")
        return None

def join_names(items: List[Dict[str, Any]], key="name", sep="|") -> str:
    return sep.join(str(it.get(key)) for it in (items or []) if it.get(key))

def find_latest_run(results_dir: Path) -> Optional[Path]:
    if not results_dir.exists(): return None
    runs = [p for p in results_dir.iterdir() if p.is_dir() and p.name.lower().startswith("tmdb_download_")]
    if not runs: return None
    runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return runs[0]

def movies_to_csv(movies_dir: Path, out_csv: Path) -> int:
    rows: List[Dict[str, Any]] = []
    for jf in sorted(movies_dir.glob("*.json")):
        d = load_json_safe(jf)
        if not d:
            continue
        prov = d.get("_providers") or {}
        rows.append({
            "id": d.get("id"),
            "title": d.get("title") or d.get("original_title"),
            "original_title": d.get("original_title"),
            "release_date": d.get("release_date"),
            "runtime": d.get("runtime"),
            "status": d.get("status"),
            "genres": join_names(d.get("genres")),
            "spoken_languages": join_names(d.get("spoken_languages")),
            "production_companies": join_names(d.get("production_companies")),
            "vote_average": d.get("vote_average"),
            "vote_count": d.get("vote_count"),
            "popularity": d.get("popularity"),
            "imdb_id": (d.get("external_ids") or {}).get("imdb_id") or d.get("imdb_id"),
            "homepage": d.get("homepage") or "",
            "poster_url": d.get("_poster_url") or "",
            "backdrop_url": d.get("_backdrop_url") or "",
            f"providers_flatrate_{REGION}": prov.get(f"providers_flatrate_{REGION}", ""),
            f"providers_rent_{REGION}":     prov.get(f"providers_rent_{REGION}", ""),
            f"providers_buy_{REGION}":      prov.get(f"providers_buy_{REGION}", ""),
            f"providers_ads_{REGION}":      prov.get(f"providers_ads_{REGION}", ""),
            f"providers_link_{REGION}":     prov.get(f"providers_link_{REGION}", ""),
        })
    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(rows[0].keys()) if rows else [
                "id","title","original_title","release_date","runtime","status",
                "genres","spoken_languages","production_companies","vote_average","vote_count","popularity",
                "imdb_id","homepage","poster_url","backdrop_url",
                f"providers_flatrate_{REGION}",f"providers_rent_{REGION}",f"providers_buy_{REGION}",f"providers_ads_{REGION}",f"providers_link_{REGION}"
            ]
        )
        writer.writeheader()
        for r in rows: writer.writerow(r)
    return len(rows)

def tv_shows_to_csv(tv_dir: Path, out_csv: Path) -> int:
    rows: List[Dict[str, Any]] = []
    for show_dir in sorted(tv_dir.glob("*")):
        if not show_dir.is_dir(): continue
        show_meta = next(show_dir.glob("show_*.json"), None)
        if not show_meta: continue
        d = load_json_safe(show_meta)
        if not d: continue
        prov = d.get("_providers") or {}
        rows.append({
            "id": d.get("id"),
            "name": d.get("name") or d.get("original_name"),
            "original_name": d.get("original_name"),
            "first_air_date": d.get("first_air_date"),
            "last_air_date": d.get("last_air_date"),
            "number_of_seasons": d.get("number_of_seasons"),
            "number_of_episodes": d.get("number_of_episodes"),
            "status": d.get("status"),
            "in_production": d.get("in_production"),
            "genres": join_names(d.get("genres")),
            "networks": join_names(d.get("networks")),
            "origin_country": "|".join(d.get("origin_country") or []),
            "languages": "|".join(d.get("languages") or []),
            "vote_average": d.get("vote_average"),
            "vote_count": d.get("vote_count"),
            "popularity": d.get("popularity"),
            "homepage": d.get("homepage") or "",
            "poster_url": d.get("_poster_url") or "",
            "backdrop_url": d.get("_backdrop_url") or "",
            f"providers_flatrate_{REGION}": prov.get(f"providers_flatrate_{REGION}", ""),
            f"providers_rent_{REGION}":     prov.get(f"providers_rent_{REGION}", ""),
            f"providers_buy_{REGION}":      prov.get(f"providers_buy_{REGION}", ""),
            f"providers_ads_{REGION}":      prov.get(f"providers_ads_{REGION}", ""),
            f"providers_link_{REGION}":     prov.get(f"providers_link_{REGION}", ""),
        })
    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(rows[0].keys()) if rows else [
                "id","name","original_name","first_air_date","last_air_date","number_of_seasons","number_of_episodes",
                "status","in_production","genres","networks","origin_country","languages","vote_average","vote_count","popularity",
                "homepage","poster_url","backdrop_url",
                f"providers_flatrate_{REGION}",f"providers_rent_{REGION}",f"providers_buy_{REGION}",f"providers_ads_{REGION}",f"providers_link_{REGION}"
            ]
        )
        writer.writeheader()
        for r in rows: writer.writerow(r)
    return len(rows)

def tv_seasons_to_csv(tv_dir: Path, out_csv: Path) -> int:
    rows: List[Dict[str, Any]] = []
    for show_dir in sorted(tv_dir.glob("*")):
        if not show_dir.is_dir(): continue
        show_meta = next(show_dir.glob("show_*.json"), None)
        base = load_json_safe(show_meta) if show_meta else {}
        show_id = base.get("id")
        show_name = base.get("name") or base.get("original_name")

        seasons_idx = show_dir / "seasons_index.json"
        s_data = load_json_safe(seasons_idx) or []
        for s in s_data:
            rows.append({
                "show_id": show_id,
                "show_name": show_name,
                "season_number": s.get("season_number"),
                "name": s.get("name"),
                "air_date": s.get("air_date"),
                "episode_count": s.get("episode_count"),
                "season_poster_url": s.get("_season_poster_url") or "",
            })
    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(rows[0].keys()) if rows else [
                "show_id","show_name","season_number","name","air_date","episode_count","season_poster_url"
            ]
        )
        writer.writeheader()
        for r in rows: writer.writerow(r)
    return len(rows)

def tv_episodes_to_csv(tv_dir: Path, out_csv: Path) -> int:
    rows: List[Dict[str, Any]] = []
    for show_dir in sorted(tv_dir.glob("*")):
        if not show_dir.is_dir(): continue
        show_meta = next(show_dir.glob("show_*.json"), None)
        base = load_json_safe(show_meta) if show_meta else {}
        show_id = base.get("id")
        show_name = base.get("name") or base.get("original_name")

        for season_file in sorted(show_dir.glob("season_*.json")):
            sdata = load_json_safe(season_file) or {}
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
                    "episode_still_url": ep.get("_still_url") or "",
                })
    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(rows[0].keys()) if rows else [
                "show_id","show_name","season_number","episode_number","name","air_date","runtime","vote_average","vote_count","episode_still_url"
            ]
        )
        writer.writeheader()
        for r in rows: writer.writerow(r)
    return len(rows)

# NEW: build one combined CSV with all rows (movies, shows, seasons, episodes)
def combined_csv(project_root: Path, rd: Path, out_csv: Path) -> int:
    """
    Build one CSV with all rows (movies, shows, seasons, episodes).
    Columns adapt; unused fields left blank per row_type.
    Row order: movies, shows, seasons, then episodes grouped by show/season.
    """
    rows: List[Dict[str, Any]] = []

    # Movies
    for jf in sorted((rd / "movies").glob("*.json")):
        d = load_json_safe(jf)
        if not d:
            continue
        prov = d.get("_providers") or {}
        rows.append({
            "row_type": "movie",
            "show_id": "", "show_name": "", "season_number": "", "episode_number": "",
            "id": d.get("id"),
            "title": d.get("title") or d.get("original_title"),
            "name": "",
            "air_date": d.get("release_date"),
            "runtime": d.get("runtime"),
            "status": d.get("status"),
            "genres": join_names(d.get("genres")),
            "vote_average": d.get("vote_average"),
            "vote_count": d.get("vote_count"),
            "popularity": d.get("popularity"),
            "homepage": d.get("homepage") or "",
            "poster_url": d.get("_poster_url") or "",
            "backdrop_url": d.get("_backdrop_url") or "",
            "imdb_id": (d.get("external_ids") or {}).get("imdb_id") or d.get("imdb_id"),
            f"providers_flatrate_{REGION}": prov.get(f"providers_flatrate_{REGION}", ""),
            f"providers_rent_{REGION}":     prov.get(f"providers_rent_{REGION}", ""),
            f"providers_buy_{REGION}":      prov.get(f"providers_buy_{REGION}", ""),
            f"providers_ads_{REGION}":      prov.get(f"providers_ads_{REGION}", ""),
            f"providers_link_{REGION}":     prov.get(f"providers_link_{REGION}", ""),
        })

    # Shows + Seasons + Episodes
    tv_dir = rd / "tv"
    for show_dir in sorted(tv_dir.glob("*")):
        if not show_dir.is_dir(): continue
        show_meta = next(show_dir.glob("show_*.json"), None)
        d = load_json_safe(show_meta) if show_meta else None
        if not d: continue
        prov = d.get("_providers") or {}
        show_id = d.get("id")
        show_name = d.get("name") or d.get("original_name")

        rows.append({
            "row_type": "show",
            "show_id": show_id, "show_name": show_name, "season_number": "", "episode_number": "",
            "id": show_id,
            "title": "", "name": show_name,
            "air_date": d.get("first_air_date"),
            "runtime": "",
            "status": d.get("status"),
            "genres": join_names(d.get("genres")),
            "vote_average": d.get("vote_average"),
            "vote_count": d.get("vote_count"),
            "popularity": d.get("popularity"),
            "homepage": d.get("homepage") or "",
            "poster_url": d.get("_poster_url") or "",
            "backdrop_url": d.get("_backdrop_url") or "",
            "imdb_id": (d.get("external_ids") or {}).get("imdb_id") or "",
            f"providers_flatrate_{REGION}": prov.get(f"providers_flatrate_{REGION}", ""),
            f"providers_rent_{REGION}":     prov.get(f"providers_rent_{REGION}", ""),
            f"providers_buy_{REGION}":      prov.get(f"providers_buy_{REGION}", ""),
            f"providers_ads_{REGION}":      prov.get(f"providers_ads_{REGION}", ""),
            f"providers_link_{REGION}":     prov.get(f"providers_link_{REGION}", ""),
        })

        seasons_idx = load_json_safe(show_dir / "seasons_index.json") or []
        seasons_idx.sort(key=lambda s: (s.get("season_number") if isinstance(s.get("season_number"), int) else 9999))

        for s in seasons_idx:
            snum = s.get("season_number")
            rows.append({
                "row_type": "season",
                "show_id": show_id, "show_name": show_name, "season_number": snum, "episode_number": "",
                "id": f"{show_id}_S{snum}",
                "title": "", "name": s.get("name"),
                "air_date": s.get("air_date"),
                "runtime": "",
                "status": "",
                "genres": "",
                "vote_average": "", "vote_count": "", "popularity": "",
                "homepage": "",
                "poster_url": s.get("_season_poster_url") or "",
                "backdrop_url": "",
                "imdb_id": "",
                f"providers_flatrate_{REGION}": "",
                f"providers_rent_{REGION}":     "",
                f"providers_buy_{REGION}":      "",
                f"providers_ads_{REGION}":      "",
                f"providers_link_{REGION}":     "",
            })

            sf = show_dir / f"season_{int(snum):02d}.json" if isinstance(snum, int) else None
            sdata = load_json_safe(sf) if sf and sf.exists() else {}
            for ep in (sdata.get("episodes") or []):
                rows.append({
                    "row_type": "episode",
                    "show_id": show_id, "show_name": show_name,
                    "season_number": sdata.get("season_number"),
                    "episode_number": ep.get("episode_number"),
                    "id": ep.get("id"),
                    "title": "", "name": ep.get("name"),
                    "air_date": ep.get("air_date"),
                    "runtime": ep.get("runtime"),
                    "status": "",
                    "genres": "",
                    "vote_average": ep.get("vote_average"),
                    "vote_count": ep.get("vote_count"),
                    "popularity": "",
                    "homepage": "",
                    "poster_url": "",
                    "backdrop_url": "",
                    "imdb_id": "",
                    f"providers_flatrate_{REGION}": "",
                    f"providers_rent_{REGION}":     "",
                    f"providers_buy_{REGION}":      "",
                    f"providers_ads_{REGION}":      "",
                    f"providers_link_{REGION}":     "",
                    "episode_still_url": ep.get("_still_url") or "",
                })

    fieldnames = [
        "row_type",
        "show_id","show_name","season_number","episode_number",
        "id","title","name","air_date","runtime","status","genres",
        "vote_average","vote_count","popularity","homepage",
        "poster_url","backdrop_url","imdb_id",
        f"providers_flatrate_{REGION}",f"providers_rent_{REGION}",f"providers_buy_{REGION}",f"providers_ads_{REGION}",f"providers_link_{REGION}",
        "episode_still_url"
    ]
    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return len(rows)

# ======= Orchestrations =======
def mode_fetch(project_root: Path, read_token: str, api_key_v3: Optional[str], list_ids: List[int]) -> Path:
    results_dir = find_or_create_dir(project_root, "Results", "results")
    logs_dir    = find_or_create_dir(project_root, "Logs", "logs")
    run_root = results_dir / f"tmdb_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    movies_dir = ensure_dir(run_root / "movies")
    tv_dir     = ensure_dir(run_root / "tv")

    setup_logging(logs_dir, prefix="run_fetch", verbose_console=False)
    progress(f"▶ Fetching TMDB lists into: {run_root}")

    client = TMDBClient(read_token=read_token, api_key_v3=api_key_v3 or None)
    total_movies = total_tv = 0

    for list_id in list_ids:
        progress(f"• List {list_id}: fetching items")
        items = list_all_items(client, list_id)
        progress(f"  - {len(items)} items")
        for item in items:
            mt = item.get("media_type") or item.get("type")
            try:
                if mt == "movie":
                    process_movie(client, item, movies_dir); total_movies += 1
                elif mt == "tv":
                    process_tv(client, item, tv_dir); total_tv += 1
            except Exception as e:
                logging.error(f"Failed item id={item.get('id')}: {e}")
                debug(traceback.format_exc())

    index = {
        "run_started": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_root),
        "movies_saved": total_movies,
        "tv_saved": total_tv,
    }
    save_json(run_root / "index.json", index)
    progress(f"✓ Fetch complete: movies={total_movies}, tv={total_tv}")
    progress(f"  Log: {CURRENT_LOG_PATH}")
    return run_root

def mode_fix_slugs(project_root: Path, run_dir: Optional[str]) -> Path:
    results_dir = find_or_create_dir(project_root, "Results", "results")
    logs_dir    = find_or_create_dir(project_root, "Logs", "logs")
    setup_logging(logs_dir, prefix="run_fix", verbose_console=False)

    rd = Path(run_dir).resolve() if run_dir else find_latest_run(results_dir)
    if not rd or not rd.exists():
        raise FileNotFoundError("Run directory not found.")

    progress(f"▶ Cleaning & renaming TV folders in: {rd}\\tv")
    renames = rename_buggy_paths(rd / "tv")
    restored = recover_missing_json_extensions(rd / "tv")
    progress(f"✓ Cleanup complete. Renamed: {len(renames)} • Restored .json: {restored}")
    progress(f"  Log: {CURRENT_LOG_PATH}")
    return rd

def mode_merge(project_root: Path, run_dir: Optional[str]) -> Path:
    results_dir = find_or_create_dir(project_root, "Results", "results")
    logs_dir    = find_or_create_dir(project_root, "Logs", "logs")
    setup_logging(logs_dir, prefix="run_merge", verbose_console=False)

    rd = Path(run_dir).resolve() if run_dir else find_latest_run(results_dir)
    if not rd or not rd.exists():
        raise FileNotFoundError("Run directory not found.")

    # If user skipped Fix step, at least try to restore missing .json so CSVs won't be empty
    recover_missing_json_extensions(rd / "tv")

    csv_run_dir = ensure_dir(rd / "csv")
    outputs_dir = ensure_dir(project_root / "Outputs" / "csv")

    progress(f"▶ Merging JSON → CSV from: {rd}")
    m = movies_to_csv(rd / "movies", csv_run_dir / "movies.csv")
    s = tv_shows_to_csv(rd / "tv", csv_run_dir / "tv_shows.csv")
    t = tv_seasons_to_csv(rd / "tv", csv_run_dir / "tv_seasons.csv")
    e = tv_episodes_to_csv(rd / "tv", csv_run_dir / "tv_episodes.csv")

    # one combined CSV with all rows
    combined_count = combined_csv(project_root, rd, csv_run_dir / "combined_all_rows.csv")

    # Copy to stable Outputs\csv
    for name in ("movies.csv","tv_shows.csv","tv_seasons.csv","tv_episodes.csv","combined_all_rows.csv"):
        try:
            (outputs_dir / name).write_text((csv_run_dir / name).read_text(encoding="utf-8"), encoding="utf-8")
        except Exception as ex:
            logging.warning(f"Copy {name} -> Outputs\\csv failed: {ex}")

    progress(f"✓ Merge complete. Rows → movies:{m}, shows:{s}, seasons:{t}, episodes:{e}, combined:{combined_count}")
    progress(f"  CSV: {csv_run_dir}  and  {outputs_dir}")
    progress(f"  Log: {CURRENT_LOG_PATH}")
    return rd

def mode_all(project_root: Path, read_token: str, api_key_v3: Optional[str], list_ids: List[int]) -> Path:
    rd = mode_fetch(project_root, read_token, api_key_v3, list_ids)
    mode_fix_slugs(project_root, str(rd))
    mode_merge(project_root, str(rd))
    return rd

# ======= Background workers (QThread) =======
class WorkerSignals(QtCore.QObject):
    progress = QtCore.pyqtSignal(int)       # 0..100
    message  = QtCore.pyqtSignal(str)       # status lines
    done     = QtCore.pyqtSignal(Path)      # run_dir

class BaseWorker(QtCore.QThread):
    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()
    def emit_progress(self, p:int):
        self.signals.progress.emit(max(0, min(100, p)))
    def say(self, msg:str):
        self.signals.message.emit(msg)

class FetchWorker(BaseWorker):
    def __init__(self, project_root: Path, read_token: str, api_key_v3: Optional[str], list_ids: List[int]):
        super().__init__()
        self.root=project_root; self.tok=read_token; self.key=api_key_v3; self.lists=list_ids
    def run(self):
        try:
            self.say("Starting Fetch…"); self.emit_progress(5)
            rd = mode_fetch(self.root, self.tok, self.key, self.lists)
            self.say("Completed Fetch"); self.emit_progress(100)
            self.signals.done.emit(rd)
        except Exception as e:
            logging.error(f"Worker error (Fetch): {e}")
            self.say(f"ERROR: {e}")
            self.emit_progress(100)
            self.signals.done.emit(self.root)

class FixWorker(BaseWorker):
    def __init__(self, project_root: Path, run_dir: Optional[str]):
        super().__init__()
        self.root=project_root; self.rd=run_dir
    def run(self):
        try:
            self.say("Starting Clean…"); self.emit_progress(10)
            rd = mode_fix_slugs(self.root, self.rd)
            self.say("Completed Clean"); self.emit_progress(100)
            self.signals.done.emit(rd)
        except Exception as e:
            logging.error(f"Worker error (Fix): {e}")
            self.say(f"ERROR: {e}")
            self.emit_progress(100)
            self.signals.done.emit(self.root)

class MergeWorker(BaseWorker):
    def __init__(self, project_root: Path, run_dir: Optional[str]):
        super().__init__()
        self.root=project_root; self.rd=run_dir
    def run(self):
        try:
            self.say("Starting Merge…"); self.emit_progress(10)
            rd = mode_merge(self.root, self.rd)
            self.say("Completed Merge"); self.emit_progress(100)
            self.signals.done.emit(rd)
        except Exception as e:
            logging.error(f"Worker error (Merge): {e}")
            self.say(f"ERROR: {e}")
            self.emit_progress(100)
            self.signals.done.emit(self.root)

class AllWorker(BaseWorker):
    def __init__(self, project_root: Path, read_token: str, api_key_v3: Optional[str], list_ids: List[int]):
        super().__init__()
        self.root=project_root; self.tok=read_token; self.key=api_key_v3; self.lists=list_ids
    def run(self):
        try:
            self.say("Starting All (Fetch → Clean → CSV)…"); self.emit_progress(5)
            rd = mode_fetch(self.root, self.tok, self.key, self.lists); self.emit_progress(40)
            self.say("Completed Fetch")
            mode_fix_slugs(self.root, str(rd)); self.emit_progress(65)
            self.say("Completed Clean")
            mode_merge(self.root, str(rd)); self.emit_progress(100)
            self.say("Completed All (Fetch → Clean → CSV)")
            self.signals.done.emit(rd)
        except Exception as e:
            logging.error(f"Worker error (All): {e}")
            self.say(f"ERROR: {e}")
            self.emit_progress(100)
            self.signals.done.emit(self.root)

# ======= GUI widgets & styling =======
class EyeToggle(QtWidgets.QToolButton):
    """Small eye button to toggle password visibility."""
    def __init__(self, line_edit: QtWidgets.QLineEdit):
        super().__init__()
        self.line_edit = line_edit
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setToolTip("Show/Hide value")
        self.setText("👁")
        self.toggled.connect(self.on_toggled)
        self.setFixedWidth(28)
    def on_toggled(self, checked: bool):
        self.line_edit.setEchoMode(QtWidgets.QLineEdit.Normal if checked else QtWidgets.QLineEdit.Password)
        self.setText("🙈" if checked else "👁")

def apply_stylesheet(app: QtWidgets.QApplication):
    s = THEME
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(s["bg"]))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(s["panel"]))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(s["panel"]))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(s["text"]))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(s["text"]))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(s["panel"]))
    app.setPalette(palette)
    app.setStyleSheet(f"""
        QWidget {{
            font-family: Segoe UI, Arial, Helvetica, sans-serif;
            color: {s["text"]};
        }}
        QGroupBox {{
            background: {s["panel"]};
            border: 1px solid {s["border"]};
            border-radius: 12px;
            margin-top: 14px;
            padding: 10px 12px 12px 12px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
            color: {s["subtext"]};
        }}
        QLineEdit, QPlainTextEdit {{
            border: 1px solid {s["border"]};
            border-radius: 10px;
            padding: 8px 10px;
            background: {s["panel"]};
        }}
        QPushButton {{
            background: {s["primary"]};
            color: white;
            border: none;
            border-radius: 12px;
            padding: 10px 14px;
        }}
        QPushButton:hover {{ background: {s["primary2"]}; }}
        QPushButton:disabled {{ background: #9BB5AA; }}
        QToolButton {{
            border: 1px solid {s["border"]};
            border-radius: 8px;
            background: {s["panel"]};
        }}
        QMenuBar {{ background: {s["panel"]}; border: 0; }}
        QMenuBar::item:selected {{ background: {s["accent"]}; color: white; border-radius: 6px; }}
        QStatusBar {{ background: {s["panel"]}; color: {s["subtext"]}; }}
    """)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TMDB List Grabber — Andrew")
        self.setMinimumWidth(840)

        # Keep strong refs to workers (prevents QThread destruction while running)
        self._workers: List[QtCore.QThread] = []

        # Create an app log immediately
        project_root_default = resolve_project_root(Path(__file__).resolve(), None)
        logs_dir = find_or_create_dir(project_root_default, "Logs", "logs")
        setup_logging(logs_dir, prefix="app", verbose_console=False)

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout(central); v.setContentsMargins(14, 12, 14, 12); v.setSpacing(12)

        # Menubar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        exit_act = QtWidgets.QAction("E&xit", self); exit_act.triggered.connect(self.close); file_menu.addAction(exit_act)
        help_menu = menubar.addMenu("&Help")
        about_act = QtWidgets.QAction("&About", self); help_act = QtWidgets.QAction("&Help", self)
        about_act.triggered.connect(self.show_about); help_act.triggered.connect(self.show_help)
        help_menu.addAction(about_act); help_menu.addAction(help_act)

        # Project & creds
        grp = QtWidgets.QGroupBox("Project & Credentials"); v.addWidget(grp)
        g = QtWidgets.QGridLayout(grp); g.setHorizontalSpacing(10); g.setVerticalSpacing(8)

        r = 0
        g.addWidget(QtWidgets.QLabel("Project root folder:"), r, 0)
        self.root_input = QtWidgets.QLineEdit(str(project_root_default))
        self.root_btn   = QtWidgets.QPushButton("Browse…"); self.root_btn.clicked.connect(self.pick_root)
        g.addWidget(self.root_input, r, 1); g.addWidget(self.root_btn, r, 2)

        r += 1
        g.addWidget(QtWidgets.QLabel("TMDB Read Access Token (v4 Bearer):"), r, 0)
        self.read_input = QtWidgets.QLineEdit(os.getenv("TMDB_READ_TOKEN", PASTE_TMDB_READ_TOKEN))
        self.read_input.setEchoMode(QtWidgets.QLineEdit.Password)
        h1 = QtWidgets.QHBoxLayout(); h1.addWidget(self.read_input); h1.addWidget(EyeToggle(self.read_input))
        g.addLayout(h1, r, 1, 1, 2)
        self.read_input.setToolTip("Your TMDB v4 Read Access Token (Bearer). Use the eye to verify.")

        r += 1
        g.addWidget(QtWidgets.QLabel("TMDB API Key (v3, optional):"), r, 0)
        self.api_input = QtWidgets.QLineEdit(os.getenv("TMDB_API_KEY", PASTE_TMDB_API_KEY))
        self.api_input.setEchoMode(QtWidgets.QLineEdit.Password)
        h2 = QtWidgets.QHBoxLayout(); h2.addWidget(self.api_input); h2.addWidget(EyeToggle(self.api_input))
        g.addLayout(h2, r, 1, 1, 2)
        self.api_input.setToolTip("Optional v3 API key. Some endpoints include extra data when provided.")

        r += 1
        g.addWidget(QtWidgets.QLabel("TMDB List IDs (space or comma separated):"), r, 0)
        self.lists_input = QtWidgets.QLineEdit(" ".join(map(str, DEFAULT_LIST_IDS)))
        g.addWidget(self.lists_input, r, 1, 1, 2)

        # Actions
        grp2 = QtWidgets.QGroupBox("Actions"); v.addWidget(grp2)
        a = QtWidgets.QGridLayout(grp2)

        self.fetch_btn = QtWidgets.QPushButton("Fetch TMDB lists now")
        self.fetch_desc = QtWidgets.QLabel("Downloads movies and TV shows from your lists. TV includes last & upcoming seasons.")
        self.fetch_desc.setStyleSheet(f"color:{THEME['subtext']}")
        a.addWidget(self.fetch_btn, 0, 0, 1, 1); a.addWidget(self.fetch_desc, 0, 1, 1, 2)

        self.fix_btn = QtWidgets.QPushButton("Clean & rename TV folders")
        self.fix_desc = QtWidgets.QLabel("Repairs broken underscore names (e.g., a_b_b_o_t_t → abbott). Safe rename, keeps .json.")
        self.fix_desc.setStyleSheet(f"color:{THEME['subtext']}")
        a.addWidget(self.fix_btn, 1, 0, 1, 1); a.addWidget(self.fix_desc, 1, 1, 1, 2)

        self.merge_btn = QtWidgets.QPushButton("Build CSV files from JSON")
        self.merge_desc = QtWidgets.QLabel(r"Creates movies/tv CSVs in run\\csv and Outputs\\csv.")
        self.merge_desc.setStyleSheet(f"color:{THEME['subtext']}")
        a.addWidget(self.merge_btn, 2, 0, 1, 1); a.addWidget(self.merge_desc, 2, 1, 1, 2)

        self.all_btn = QtWidgets.QPushButton("Run everything (Fetch → Clean → CSV)")
        a.addWidget(self.all_btn, 3, 0, 1, 3)

        self.fetch_btn.clicked.connect(self.on_fetch)
        self.fix_btn.clicked.connect(self.on_fix)
        self.merge_btn.clicked.connect(self.on_merge)
        self.all_btn.clicked.connect(self.on_all)

        # Status
        grp3 = QtWidgets.QGroupBox("Status"); v.addWidget(grp3)
        s = QtWidgets.QVBoxLayout(grp3)
        self.status = QtWidgets.QPlainTextEdit(); self.status.setReadOnly(True); self.status.setMinimumHeight(220)
        s.addWidget(self.status)

        # Progress + clickable links
        self.prog = QtWidgets.QProgressBar(); self.prog.setRange(0, 100); self.prog.setValue(0); v.addWidget(self.prog)
        self.links = QtWidgets.QLabel(""); self.links.setOpenExternalLinks(True); v.addWidget(self.links)

        self.statusBar().showMessage("Ready")

    # Helpers
    def say(self, msg: str):
        self.status.appendPlainText(msg)
        progress(msg)
        self.status.verticalScrollBar().setValue(self.status.verticalScrollBar().maximum())
        self.statusBar().showMessage(msg[:120])

    def pick_root(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Project Root", self.root_input.text())
        if d: self.root_input.setText(d)

    def parse_lists(self) -> List[int]:
        raw = self.lists_input.text().replace(",", " ").split()
        out: List[int] = []
        for t in raw:
            try: out.append(int(t))
            except: pass
        return out or DEFAULT_LIST_IDS

    def set_running(self, running: bool):
        for btn in (self.fetch_btn, self.fix_btn, self.merge_btn, self.all_btn, self.root_btn):
            btn.setEnabled(False if running else True)
        self.prog.setValue(0 if not running else 5)
        self.statusBar().showMessage("Working…" if running else "Ready")

    def show_done_dialog(self, title: str, rd: Path):
        QtWidgets.QMessageBox.information(self, title, f"{title}\n\nRun folder:\n{rd}\n\nYou can click the links below to open folders.")

    def connect_worker(self, worker, title_after: str, show_csv_links: bool):
        # keep a strong reference so thread isn't GC'd mid-run
        self._workers.append(worker)

        self.set_running(True)
        worker.signals.progress.connect(self.prog.setValue)
        worker.signals.message.connect(self.say)

        def finished(rd: Path):
            # graceful cleanup
            try:
                if worker.isRunning():
                    worker.quit()
                    worker.wait(3000)
            except Exception:
                pass
            try:
                self._workers.remove(worker)
            except ValueError:
                pass

            # update UI
            self.prog.setValue(100)
            self.set_running(False)
            self.say(f"Completed: {title_after}")

            rd = Path(rd)
            csv_dir = rd / "csv"
            out_dir = Path(self.root_input.text()).resolve() / "Outputs" / "csv"
            link_rd  = f'<a href="file:///{rd}">{rd}</a>'
            link_csv = f'<a href="file:///{csv_dir}">{csv_dir}</a>'
            link_out = f'<a href="file:///{out_dir}">{out_dir}</a>'
            if show_csv_links:
                self.links.setText(f"Run: {link_rd} • CSV: {link_csv} • Outputs: {link_out}")
            else:
                self.links.setText(f"Run: {link_rd}")
            self.show_done_dialog(title_after, rd)

        # IMPORTANT: only our custom 'done' signal to avoid double dialogs & wrong path
        worker.signals.done.connect(finished)
        worker.start()

    # Menus
    def show_about(self):
        QtWidgets.QMessageBox.information(
            self, "About",
            "TMDB List Grabber\n• Fetch TMDB lists\n• Clean & rename folders\n• Merge JSON → CSV\n\nQuiet UI; detailed logs for support."
        )

    def show_help(self):
        QtWidgets.QMessageBox.information(
            self, "Help",
            "1) Select your project root and enter credentials (use the eye to verify).\n"
            "2) Click an action:\n"
            "   • Fetch TMDB lists now – downloads movies and TV data.\n"
            "   • Clean & rename TV folders – fixes broken underscore names safely.\n"
            "   • Build CSV files from JSON – creates movies/tv CSVs.\n"
            "3) Watch the progress bar; a completion dialog appears when finished.\n"
            "4) Click the links under the progress bar to open folders."
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # Prevent “QThread destroyed” by stopping workers first
        if any(w.isRunning() for w in self._workers):
            self.statusBar().showMessage("Stopping background tasks…")
            for w in list(self._workers):
                try:
                    if w.isRunning():
                        w.requestInterruption()
                        w.quit()
                        w.wait(3000)
                except Exception:
                    pass
            self._workers.clear()
        event.accept()

    # Actions (spawn background workers)
    def on_fetch(self):
        try:
            w = FetchWorker(Path(self.root_input.text()).resolve(),
                            self.read_input.text().strip(),
                            (self.api_input.text().strip() or None),
                            self.parse_lists())
            self.connect_worker(w, "Fetch complete", show_csv_links=False)
        except Exception as e:
            self.say(f"ERROR: {e}\nSee log: {CURRENT_LOG_PATH}")

    def on_fix(self):
        try:
            w = FixWorker(Path(self.root_input.text()).resolve(), None)
            self.connect_worker(w, "Clean complete", show_csv_links=False)
        except Exception as e:
            self.say(f"ERROR: {e}\nSee log: {CURRENT_LOG_PATH}")

    def on_merge(self):
        try:
            w = MergeWorker(Path(self.root_input.text()).resolve(), None)
            self.connect_worker(w, "CSV build complete", show_csv_links=True)
        except Exception as e:
            self.say(f"ERROR: {e}\nSee log: {CURRENT_LOG_PATH}")

    def on_all(self):
        try:
            w = AllWorker(Path(self.root_input.text()).resolve(),
                          self.read_input.text().strip(),
                          (self.api_input.text().strip() or None),
                          self.parse_lists())
            # show CSV links at end of "all"
            self.connect_worker(w, "All steps complete", show_csv_links=True)
        except Exception as e:
            self.say(f"ERROR: {e}\nSee log: {CURRENT_LOG_PATH}")

# ======= CLI =======
def run_cli():
    ap = argparse.ArgumentParser(description="TMDB List Grabber (quiet console, full log; GUI + CLI)")
    ap.add_argument("--mode", choices=["fetch","fix-slugs","merge","all","gui"], default="gui")
    ap.add_argument("--root", help="Project root (defaults to parent of Scripts/ or script folder).")
    ap.add_argument("--lists", nargs="*", type=int, help="List IDs (default from script).")
    ap.add_argument("--read_token", help="TMDB v4 Read Token (Bearer).")
    ap.add_argument("--api_key", help="TMDB v3 API key (optional).")
    ap.add_argument("--run_dir", help=r"Specific tmdb_download_... folder for fix/merge modes (e.g., C:\path\to\Results\tmdb_download_YYYYMMDD_HHMMSS)")
    ap.add_argument("--verbose", action="store_true", help="Chatty console (INFO) instead of quiet.")
    args = ap.parse_args()

    script_file = Path(__file__).resolve()
    project_root = resolve_project_root(script_file, args.root)

    read_token = (args.read_token or PASTE_TMDB_READ_TOKEN or os.getenv("TMDB_READ_TOKEN", "")).strip()
    api_key_v3 = (args.api_key or PASTE_TMDB_API_KEY or os.getenv("TMDB_API_KEY", "")).strip() or None
    list_ids = args.lists if args.lists else DEFAULT_LIST_IDS

    if args.mode == "gui":
        if not PYQT_AVAILABLE:
            print("PyQt5 not available. Install with: pip install PyQt5")
            sys.exit(2)
        app = QtWidgets.QApplication(sys.argv)
        apply_stylesheet(app)
        w = MainWindow()
        w.show()
        sys.exit(app.exec_())

    # CLI modes (quiet by default)
    logs_dir = find_or_create_dir(project_root, "Logs", "logs")
    setup_logging(logs_dir, prefix=f"run_{args.mode}", verbose_console=args.verbose)

    try:
        if args.mode in ("fetch","all") and not read_token:
            progress("ERROR: TMDB Read Token is required for fetch/all. Provide --read_token or set TMDB_READ_TOKEN.")
            progress(f"See log: {CURRENT_LOG_PATH}")
            sys.exit(2)

        if args.mode == "fetch":
            run_dir = mode_fetch(project_root, read_token, api_key_v3, list_ids)
            progress(f"Run dir: {run_dir}")

        elif args.mode == "fix-slugs":
            run_dir = mode_fix_slugs(project_root, args.run_dir)
            progress(f"Run dir: {run_dir}")

        elif args.mode == "merge":
            run_dir = mode_merge(project_root, args.run_dir)
            progress(f"Run dir: {run_dir}")

        elif args.mode == "all":
            run_dir = mode_all(project_root, read_token, api_key_v3, list_ids)
            progress(f"Run dir: {run_dir}")

    except Exception as e:
        logging.error(f"Unhandled error: {e}")
        debug(traceback.format_exc())
        progress(f"ERROR: {e}")
        if CURRENT_LOG_PATH:
            progress(f"See log: {CURRENT_LOG_PATH}")
        sys.exit(1)

if __name__ == "__main__":
    run_cli()
