import csv
import json
import os
import zipfile
from glob import glob
from urllib.request import urlretrieve
from datetime import datetime, date
from toss_engine import TossParticleFilter, TossWinnerOnlyPF


pf = TossParticleFilter()
pf.load()

# Winner-only model (for internet datasets without call/result)
win_pf = TossWinnerOnlyPF()
win_pf.load()


def _norm_call(x: str) -> str:
    x = (x or "").strip().upper()
    if x not in {"H", "T"}:
        raise ValueError("call/result must be 'H' or 'T'")
    return x


def _parse_date(s: str) -> date:
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    raise ValueError("Unrecognized date format. Use YYYY-MM-DD or DD-MM-YYYY or DD/MM/YYYY.")


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
    # Scale drift roughly per week; at least 1.0 per update
    return max(1.0, diff_days / 7.0)


def add_match():
    line = input("Enter match as CSV: date,venue,location,captain,call(H/T),result(H/T): ")
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) != 6:
        print("Invalid input. Expected exactly 6 comma-separated values.")
        return
    dt, venue, location, captain, call, result = parts
    try:
        call = _norm_call(call)
        result = _norm_call(result)
    except ValueError as e:
        print(str(e))
        return

    # Time-scaled drift step
    scale = _step_scale(pf.last_date, dt)
    pf.predict_step(step_scale=scale)
    pf.update(venue, captain, call, result, location=location or None)
    pf.last_date = dt

    pf.save()
    print("Match added and model updated.\n")


def import_csv():
    path = input("CSV path with rows: date,venue,location,captain,call,result (header optional): ").strip()
    added = 0
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = None
            for row in reader:
                if not row:
                    continue
                if header is None:
                    lower = [c.strip().lower() for c in row]
                    if set(["date", "venue", "captain", "call", "result"]).issubset(set(lower)) or \
                       set(["date", "venue", "location", "captain", "call", "result"]).issubset(set(lower)):
                        header = lower
                        continue
                    else:
                        header = []  # no header; treat current row as data

                if header:
                    cols = {name: None for name in ["date", "venue", "location", "captain", "call", "result"]}
                    for i, val in enumerate(row):
                        key = header[i] if i < len(header) else None
                        if key in cols:
                            cols[key] = val.strip()
                    dt = cols["date"] or ""
                    venue = cols["venue"] or ""
                    location = cols["location"] or ""
                    captain = cols["captain"] or ""
                    call = cols["call"] or ""
                    result = cols["result"] or ""
                else:
                    # no header; expect 6 or 5 columns
                    vals = [c.strip() for c in row]
                    if len(vals) >= 6:
                        dt, venue, location, captain, call, result = vals[:6]
                    elif len(vals) >= 5:
                        dt, venue, captain, call, result = vals[:5]
                        location = ""
                    else:
                        continue

                try:
                    call = _norm_call(call)
                    result = _norm_call(result)
                except ValueError:
                    continue

                scale = _step_scale(pf.last_date, dt)
                pf.predict_step(step_scale=scale)
                pf.update(venue, captain, call, result, location=location or None)
                pf.last_date = dt
                added += 1
        pf.save()
        print(f"Imported {added} matches.\n")
    except FileNotFoundError:
        print("File not found.")


def predict_single():
    venue = input("Venue: ").strip()
    location = input("Location (city/country, optional): ").strip()
    captain = input("Calling Captain: ").strip()

    p, unc, callprob = pf.predict(venue, captain, location=location or None)

    print("\n--- Prediction (Caller Known) ---")
    print(f"Captain Heads Calling Tendency: {callprob*100:.1f}%")
    print(f"Toss Win Probability: {p*100:.2f}%")
    print(f"Uncertainty (std): ±{unc*100:.2f}%\n")


def predict_two():
    venue = input("Venue: ").strip()
    location = input("Location (city/country, optional): ").strip()
    captain_a = input("Captain A (Team A): ").strip()
    captain_b = input("Captain B (Team B): ").strip()
    caller = input("Caller (A/B/captain name or blank if unknown): ").strip()

    res = pf.predict_two_captains(venue, captain_a, captain_b, caller=caller or None, location=location or None)
    print("\n--- Two-captain Toss Win Probabilities ---")
    print(f"If A calls: P(A wins) = {res['A_calls_A_wins']*100:.2f}%  |  P(B wins) = {(1-res['A_calls_A_wins'])*100:.2f}%")
    print(f"If B calls: P(B wins) = {res['B_calls_B_wins']*100:.2f}%  |  P(A wins) = {(1-res['B_calls_B_wins'])*100:.2f}%")
    print(f"Overall (caller unknown): P(A) = {res['A_overall']*100:.2f}%, P(B) = {res['B_overall']*100:.2f}%\n")


def show_help():
    print(
        """
Commands:
  add           - Add a single match (date,venue,location,captain,call,result)
  import_csv    - Bulk import historical matches from CSV (supports optional 'location' column)
  predict       - Predict toss win probability for a specific calling captain (optional location)
  predict_two   - Predict toss win probabilities for two captains (caller known/unknown, optional location)
  win_import_dir   - Import Cricsheet JSON from a local directory to train winner-only model
  win_import_auto  - Auto-download Cricsheet T20I JSON pack and import into winner-only model
  win_predict_two  - Predict toss winner probability between two teams (winner-only model)
  help          - Show this help
  exit          - Save and quit
        """.strip()
    )


if __name__ == "__main__":
    show_help()
    while True:
        cmd = input("Type command (help/add/import_csv/predict/predict_two/win_import_dir/win_import_auto/win_predict_two/exit): ").strip().lower()
        if cmd == "add":
            add_match()
        elif cmd == "import_csv":
            import_csv()
        elif cmd == "predict":
            predict_single()
        elif cmd == "predict_two":
            predict_two()
        elif cmd == "win_import_dir":
            path = input("Directory containing Cricsheet JSON files: ").strip()
            def _import_cricsheet_dir(d: str) -> int:
                added = 0
                files = glob(os.path.join(d, "**", "*.json"), recursive=True)
                for fp in files:
                    try:
                        with open(fp, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        info = data.get("info", {})
                        match_type = (info.get("match_type") or info.get("match_type_number") or "").lower()
                        # Accept T20I/T20
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
                        # Time-scaled drift
                        scale = _step_scale(win_pf.last_date, dt)
                        win_pf.predict_step(step_scale=scale)
                        win_pf.update(venue, team_a, team_b, winner, location=location or None)
                        win_pf.last_date = dt
                        added += 1
                    except Exception:
                        continue
                win_pf.save()
                return added
            if os.path.isdir(path):
                n = _import_cricsheet_dir(path)
                print(f"Imported {n} matches into winner-only model.\n")
            else:
                print("Not a directory.")
        elif cmd == "win_import_auto":
            # Attempt to download Cricsheet T20I JSON pack and import
            url = input("Cricsheet T20I JSON zip URL (default https://cricsheet.org/downloads/t20is_json.zip): ").strip() or "https://cricsheet.org/downloads/t20is_json.zip"
            out_dir = os.path.join("data", "cricsheet_t20i_json")
            os.makedirs("data", exist_ok=True)
            zip_path = os.path.join("data", "t20is_json.zip")
            try:
                print(f"Downloading {url} ...")
                urlretrieve(url, zip_path)
                print("Download complete. Extracting...")
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(out_dir)
                print("Extracted. Importing...")
                # Reuse directory importer
                added = 0
                files = glob(os.path.join(out_dir, "**", "*.json"), recursive=True)
                for fp in files:
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
                    except Exception:
                        continue
                win_pf.save()
                print(f"Imported {added} matches into winner-only model.\n")
            except Exception as e:
                print(f"Auto-download failed: {e}")
        elif cmd == "win_predict_two":
            venue = input("Venue: ").strip()
            location = input("Location (city/country, optional): ").strip()
            team_a = input("Team A: ").strip()
            team_b = input("Team B: ").strip()
            pA, stdA = win_pf.predict_two(venue, team_a, team_b, location=location or None)
            print("\n--- Winner-only Toss Prediction ---")
            print(f"P({team_a} wins toss): {pA*100:.2f}%  |  P({team_b} wins toss): {(1-pA)*100:.2f}%")
            print(f"Uncertainty (std on P({team_a})): ±{stdA*100:.2f}%\n")
        elif cmd == "help":
            show_help()
        elif cmd == "exit":
            pf.save()
            win_pf.save()
            break
        else:
            print("Unknown command. Type 'help' for options.")
