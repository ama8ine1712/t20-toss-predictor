import os
import json
import zipfile
from glob import glob
from urllib.request import urlretrieve, Request, urlopen
from datetime import datetime, date

from toss_engine import TossWinnerOnlyPF


CANDIDATE_URLS = [
    # Common Cricsheet JSON packs for T20/T20I; try in order until one works
    "https://cricsheet.org/downloads/t20is_json.zip",   # T20 Internationals JSON
    "https://cricsheet.org/downloads/t20s_json.zip",    # All T20s JSON
    "https://cricsheet.org/downloads/t20s_male_json.zip",
]
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "t20is_json.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "cricsheet_t20i_json")


def _parse_date(s: str) -> date:
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    # Fallback: return today to avoid zero scale
    return datetime.utcnow().date()


def _step_scale(prev_date_str: str | None, curr_date_str: str) -> float:
    try:
        curr = _parse_date(curr_date_str)
    except Exception:
        return 1.0
    if not prev_date_str:
        return 1.0
    try:
        prev = _parse_date(prev_date_str)
    except Exception:
        return 1.0
    diff_days = max(0, (curr - prev).days)
    return max(1.0, diff_days / 7.0)


def _url_exists(url: str) -> bool:
    try:
        req = Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=30) as resp:  # nosec - fetching public dataset
            return 200 <= resp.status < 400
    except Exception:
        return False


def download_cricsheet_zip(zip_path: str = ZIP_PATH) -> None:
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
        return
    last_err: Exception | None = None
    for url in CANDIDATE_URLS:
        try:
            if not _url_exists(url):
                continue
            urlretrieve(url, zip_path)
            return
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError("No valid Cricsheet T20 JSON URL reachable.")


def extract_zip(zip_path: str = ZIP_PATH, out_dir: str = EXTRACT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out_dir)


def import_cricsheet_dir(win_pf: TossWinnerOnlyPF, directory: str) -> int:
    added = 0
    files = glob(os.path.join(directory, "**", "*.json"), recursive=True)
    for i, fp in enumerate(files, 1):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            info = data.get("info", {})
            match_type = (info.get("match_type") or info.get("match_type_number") or "").lower()
            if "t20" not in str(match_type):
                continue
            teams = info.get("teams") or []
            if not teams or len(teams) < 2:
                continue
            toss = info.get("toss") or {}
            winner = toss.get("winner")
            if not winner:
                continue
            dt_list = info.get("dates") or []
            dt = str(dt_list[0]) if dt_list else ""
            venue = info.get("venue") or ""
            location = info.get("city") or info.get("venue_city") or ""
            team_a, team_b = teams[0], teams[1]

            scale = _step_scale(win_pf.last_date, dt)
            win_pf.predict_step(step_scale=scale)
            win_pf.update(venue, team_a, team_b, winner, location=location or None)
            win_pf.last_date = dt
            added += 1
            if i % 250 == 0:
                print(f"Processed {i} files, imported {added} matches...")
        except Exception:
            continue
    return added


def main() -> None:
    win_pf = TossWinnerOnlyPF()
    win_pf.load()
    print("Downloading Cricsheet T20I JSON pack (if not present)...")
    download_cricsheet_zip()
    print("Extracting JSON pack...")
    extract_zip()
    print("Importing matches into winner-only model...")
    added = import_cricsheet_dir(win_pf, EXTRACT_DIR)
    win_pf.save()
    print(f"Done. Imported {added} matches. Model saved to winner_data.json")


if __name__ == "__main__":
    main()

