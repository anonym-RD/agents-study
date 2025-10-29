"""
Fetch and parse GitHub commit diffs (using the REST API).
1) Single commit mode: we give owner, repo, sha
2) Commit window mode: we give a commit (owner, repo, sha) + a window (e.g. 1w/7d/2m)
 and we collect commits in that repo from the commit date forward.
"""
from functools import lru_cache
from typing import Dict, List, Optional, Generator
from datetime import datetime, timedelta
from dataclasses import dataclass
import requests
import os
import re
import time
import argparse
import csv
import json

def read_gh_token() -> str:
    return os.getenv("GITHUB_TOKEN", "").strip()

def _gh_headers(token: Optional[str] = None):
    if token is None:
        token = read_gh_token()
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json, application/vnd.github.text-match+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "spoiler-scraping-diffs/1.0",
    }

@dataclass
class Hunk:
    first_old: int                 # first line of the old file
    lines_old: int                 # number of lines covered of the old file
    first_new: int                 # first line of the new file
    lines_new: int                 # number of lines covered of the new file
    heading: Optional[str] = None  # optional trailing "@@ ... @@" from the heading

@dataclass
class Line:
    type: str                      # add / del / no_mod
    content: str                   # line text without the + / - / ' ' prefix
    line_old: Optional[int]        # line number in old file
    line_new: Optional[int]        # line number in new file


# calls REST to fetch a single commit
def fetch_commit(owner: str, repo: str, sha: str, token: Optional[str] = None):
    headers = _gh_headers(token)
    url = "https://api.github.com/repos/{owner}/{repo}/commits/{sha}".format(owner=owner, repo=repo, sha=sha)

    attempt = 0
    while attempt < 3:
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            attempt += 1
            print(f"Retrying {attempt}...")
            if attempt < 3:
                time.sleep(2 ** (attempt + 1))
            else:
                print("Failed after 3 attempts.")
                return None
    return None

# parse unified diff hunk header line into Hunk object
def parse_header(line: str):
    # unified-diff hunk headers like : @@ -a,b +c,d @@ optional_heading
    _HUNK_RE = re.compile(r'^@@\s*-(\d+)(?:,(\d+))?\s*\+(\d+)(?:,(\d+))?\s*@@(?:\s*(.*))?$')
    m = _HUNK_RE.match(line)
    if not m:
        return None
    return Hunk(
        int(m.group(1)),          # first_old
        int(m.group(2) or 1),     # lines_old (default 1)
        int(m.group(3)),          # first_new
        int(m.group(4) or 1),     # lines_new (default 1)
        (m.group(5) or "").strip() or None  # optional heading text
        )

# parse a unified diff patch text into Line
def parse_unified_patch(patch: str):
    if not patch:
        return []
    edits: List[Line] = []
    kon_old = 0                                                               # current old line number
    kon_new = 0                                                               # current new line no

    for raw_line in patch.splitlines():
        if raw_line.startswith('--- ') or raw_line.startswith('+++ '):           # skip file headers
            continue

        if raw_line.startswith('@@'):                                            # start of a new hunk
            h = parse_header(raw_line)                                           # parse positions
            if h is None:                                                        # skip if wrong
                continue
            kon_old = h.first_old                                                # reset counters per hunk
            kon_new = h.first_new
            continue

        if raw_line.startswith('+'):                                             # addition
            edits.append(Line('add', raw_line[1:], line_old=None, line_new=kon_new))
            kon_new += 1                                                         # increase new counter

        elif raw_line.startswith('-'):                                           # deletion
            edits.append(Line('del', raw_line[1:], line_old=kon_old, line_new=None))
            kon_old += 1                                                         # increase old counter

        elif raw_line.startswith(' '):                                           # unchanged
            edits.append(Line('no_mod', raw_line[1:], line_old=kon_old, line_new=kon_new))
            kon_old += 1                                                        # increase both
            kon_new += 1
    return edits

def fetch_compare_diffs(owner: str, repo: str, base_sha: str, head_sha: str, token: Optional[str] = None):
    # 2 target commit diff
    headers = _gh_headers(token)

    url = f"https://api.github.com/repos/{owner}/{repo}/compare/{base_sha}...{head_sha}"

    attempt = 0
    while attempt < 3:
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
            for f in data.get("files", []):
                patch = f.get("patch")
                f["edits"] = [e.__dict__ for e in parse_unified_patch(patch)] if patch else []
            return data
        except requests.RequestException as e:
            attempt += 1
            print(f"Retrying {attempt}...")
            if attempt < 3:
                time.sleep(2 ** (attempt + 1))
            else:
                print("Failed after 3 attempts.")
            return None
    return None


def get_commit_diffs(owner: str, repo: str, sha: str, token: Optional[str] = None):
    data = fetch_commit(owner, repo, sha, token=token)
    commit_info = {
        "repo": f"{owner}/{repo}",
        "sha": data.get("sha"),
        "html_url": data.get("html_url"),
        "author": data.get("commit", {}).get("author"),         # author metadata
        "committer": data.get("commit", {}).get("committer"),   # committer metadata
        "message": data.get("commit", {}).get("message"),       # commit message
        "stats": data.get("stats") or {},
        "files": [],
        "author_date": (data.get("commit", {}) or {}).get("author", {}).get("date"),
        "commit_date": (data.get("commit", {}) or {}).get("committer", {}).get("date"),
    }

    for f in data.get("files", []) or []:                       # changed files
        file_entry = {
            "filename": f.get("filename"),                      # path
            "status": f.get("status"),                          # modified/added/deleted/renamed
            "additions": f.get("additions"),
            "deletions": f.get("deletions"),
            "changes": f.get("changes"),
            "previous_filename": f.get("previous_filename"),
            "raw_url": f.get("raw_url"),
            "blob_url": f.get("blob_url"),
            "patch": f.get("patch"),
        }
        patch = f.get("patch")
        file_entry["edits"] = [e.__dict__ for e in parse_unified_patch(patch)] if patch else [] # serialize Line
        commit_info["files"].append(file_entry)
    return commit_info

# convert into table of rows
def convert_to_table(commit_diffs: Dict):
    rows: List[Dict] = []                                                      # result table
    base = {
        "repo": commit_diffs.get("repo"),
        "sha": commit_diffs.get("sha"),
        "author_date": commit_diffs.get("author_date"),                        # author date per row
        "commit_date": commit_diffs.get("commit_date"),                        # committer date per row
        "message": (commit_diffs.get("message") or "").splitlines()[0],        # first line summary
    }
    for f in commit_diffs.get("files", []):                                    # each file
        for e in f.get("edits", []):                                           # each edited line
            rows.append({
                **base,                                                        # copy base columns
                "filename": f.get("filename"),
                "status": f.get("status"),
                "type": e.get("type"),
                "line_old": e.get("line_old"),
                "line_new": e.get("line_new"),
                "code": e.get("content"),
            })
    return rows

# search commits for window collection
def get_last_commit_in_window(owner: str,repo: str,start_day: str,end_day: str,token: Optional[str] = None):
    headers = _gh_headers(token)
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"

    params = {
        "since": f"{start_day}T00:00:00Z",
        "until": f"{end_day}T23:59:59Z",
        "per_page": 1,
    }
    attempt = 0
    while attempt < 3:
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)

            if resp.status_code == 404:
                return None
            resp.raise_for_status()

            commits = resp.json()
            if not commits:
                return None

            last = commits[0]
            return {
                "owner": owner,
                "repo": repo,
                "sha": last.get("sha"),
                "html_url": last.get("html_url"),
                "date": (last.get("commit") or {}).get("committer", {}).get("date"),
            }
        except requests.RequestException as e:
            attempt += 1
            print(f"Retrying {attempt}...")
            if attempt < 3:
                time.sleep(2 ** (attempt + 1))
            else:
                print("Failed after 3 attempts.")
                return None
    return None

# given a commit, get its author date or committer date build a repo-scoped date range starting at the specific date
# and collect diff rows for all commits in that forward window
# Note: its only 1 day precision + initial defaults
def save_rows(owner: str, repo: str, start_commit_sha: str, window: str,
                                      date: str = "committer",
                                      sort: Optional[str] = "committer-date",
                                      order: str = "asc",
                                      limit: int = 50,
                                      token: Optional[str] = None,
                                      filter: Optional[str] = None):

    diff_rows: List[Dict] = []
    start_commit = fetch_commit(owner, repo, start_commit_sha, token=token)
    if date == "author":
        format_date = (start_commit.get("commit") or {}).get("author").get("date")
        date_key = "author-date"
    else:
        format_date = (start_commit.get("commit") or {}).get("committer").get("date")
        date_key = "committer-date"
    start_commit_date = format_date.split("T", 1)[0]              # truncate to YYYY-MM-DD

    m = re.match(r'^\s*(\d+)\s*(w|week|weeks|d|day|days|m|month|months|h|hour|hours|min|minute|minutes)\s*$', window or "", re.IGNORECASE)  # parser for window
    n = int (m.group(1))
    minutes = 1440
    hours = 24
    memo = m.group(2).lower()
    if memo.startswith('w'):
        days = 7 * n
    elif memo.startswith('d'):
        days = n
    elif memo.startswith('min'):
        days = max(1, (n + minutes - 1) // minutes)
    elif memo.startswith('h'):
        days = max(1, (n + hours - 1) // hours)
    else:
        days = 30 * n

    start_day = start_commit_date
    end_day = (datetime.strptime(start_commit_date, "%Y-%m-%d").date() + timedelta(days=days)).strftime("%Y-%m-%d")

    q_parts = [f"repo:{owner}/{repo}", f"{date_key}:{start_day}..{end_day}"]           # search query: repo + chosen date
    if filter:
        q_parts.append(filter)
    filter = " ".join(q_parts)

    for c in commit_search(filter, sort=(sort or date_key), order=order, limit=limit, token=token):
            d = get_commit_diffs(c["owner"], c["repo"], c["sha"], token=token)
            diff_rows.extend(convert_to_table(d))
    return diff_rows

# save rows to CSV file
def write_csv(rows: List[Dict], save_csv: Optional[str]):
    keys = list(rows[0].keys())                                   # CSV header from first row
    with open(save_csv, "w", newline="", encoding="utf-8") as f:  # open file
        w = csv.DictWriter(f, fieldnames=keys)                    # writer with columns
        w.writeheader()                                           # header row
        w.writerows(rows)                                         # all data
    print(f"Saved to CSV: {save_csv}")

def commit_search(query: str, sort: Optional[str], order: str,
                    limit: int, token: Optional[str] = None):
    headers = _gh_headers(token)                            # auth headers
    params = {"q": query, "per_page": 100}                  # base params
    if sort:
        params["sort"] = sort                               # e.g. author-date / committer-date
        params["order"] = order                             # asc / desc

    kon = 0                                                 # counter
    page = 1                                                # start page

    while kon < limit:                                      # paginate under limit
        params["page"] = page
        resp = requests.get("https://api.github.com/search/commits" , headers=headers, params=params) # used to find commits in a window

        if resp.status_code == 403 and "rate limit" in resp.text.lower():                                   # RETRY
            reset = int (resp.headers.get("X-RateLimit-Reset", "0"))
            wait = reset - int (time.time()) + 1
            if wait < 1:
                wait = 1
            time.sleep(min(wait, 60))
            resp = requests.get("https://api.github.com/search/commits" , headers=headers, params=params)

        resp.raise_for_status()
        items = resp.json().get("items") or []
        if not items:
            break

        for it in items:
            if kon >= limit:
                break
            repo_info = it.get("repository") or {}
            owner = (repo_info.get("owner") or {}).get("login") or ""
            repo = repo_info.get("name") or ""
            yield {"owner": owner, "repo": repo, "sha": it.get("sha"), "html_url": it.get("html_url")}
            kon += 1
        page += 1                                           # goes to next page

def _cli():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)                                                # choose one mode
    mode.add_argument("--sha")                                                                          # single commit mode
    mode.add_argument("--start_commit_sha")                                                             # commit window mode

    p.add_argument("--owner")                                                                           # owner for targeted modes
    p.add_argument("--repo")                                                                            # repo for targeted modes
    p.add_argument("--sort", default=None)
    p.add_argument("--order", default="desc", choices=["asc", "desc"])                      # asc or desc for window
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--window", default=None)
    p.add_argument("--start_commit-date", default="committer", choices=["author", "committer"])   # choose date field
    p.add_argument("--query", default=None)                                                 # extra filter
    p.add_argument("--as-json", action="store_true")                                        # store as JSON
    args = p.parse_args()                                                                                # parse CLI args

    if args.sha:                                                                                         # single-commit branch
        d = get_commit_diffs(args.owner, args.repo, args.sha)                                            # fetch commit
        if args.as_json:                                                                                 # print JSON
            print(json.dumps(d, indent=1))
        else:                                                                                            # flatten + save
            rows = convert_to_table(d)
            write_csv(rows, "saved_data/commit_diff.csv")
            print(f"Succesfully written to CSV!")

    else:                                                                                                # commit window branch
        rows = save_rows(
            owner=args.owner,
            repo=args.repo,
            start_commit_sha=args.start_commit_sha,
            window=args.window,
            date=args.start_commit_date,                                                                 # author or committer date
            sort=(args.sort or ("committer-date" if args.start_commit_date == "committer" else "author-date")),
            order=(args.order or "asc"),
            limit=args.limit,
            filter=args.query,
        )
        write_csv(rows, "saved_data/multiple_commit_diff.csv")
        print(f"Successfully written to CSV!")

if __name__ == "__main__":
    _cli()

"""
Testing examples:
1) Single commit:
python -m spoiler.scraping.diffs --owner microsoft --repo vscode-pull-request-github --sha c281f31e5e0b2f1e5e6c872e1c80a1861e8e09ec
python -m spoiler.scraping.diffs --owner microsoft --repo vscode-pull-request-github --sha c281f31e5e0b2f1e5e6c872e1c80a1861e8e09ec --as-json


2) Window commit:
python -m spoiler.scraping.diffs --start_commit_sha f4cb0b9452a508919f068a0fd0a0611ef1e35e42 --owner microsoft --repo vscode-pull-request-github --window 1w --sort committer-date --order asc --limit 50 
"""
