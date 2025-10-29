from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass
import time
import base64
import requests
import os
import re
import argparse
from datetime import datetime, timedelta
import pandas as pd

from diffs import (
    read_gh_token, _gh_headers,
    fetch_commit, get_commit_diffs,
    get_last_commit_in_window,
)

@dataclass
class Line:
    filename: str
    content: str

# we will only take into account the added and modified lines from each file
# for lines that are modified diff first remove them and then add them so they will be counted as added

def get_lines(owner: str, repo: str, sha: str, token: Optional[str]):
    diff = get_commit_diffs(owner, repo, sha, token)
    lines: List[Line] = []
    for file in diff.get("files", []):
        if file.get("status") != "deleted":
            for it in file.get("edits", []):
                if it.get("type") == "add":
                    lines.append(Line(file.get("filename", ""), it.get("content") or ""))
    return lines

def get_file_content(owner: str, repo: str, path: str, ref: str, token: Optional[str]) -> Optional[str]:
    """
    Get the file from GitHub at a specific commit and returns the file text in UTF-8.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"

    headers = _gh_headers(token)
    try:
        resp = requests.get(url, headers=headers, params={"ref": ref}, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        if data.get("type") == "file":         # make sure it is a file
            return base64.b64decode(data.get("content")).decode("utf-8", errors="replace")  # decode it because its Base64 encoded
        return None
    except requests.RequestException as e:
        return None

def compute_survival(owner: str, repo: str, init_sha: str, window: str, token: Optional[str] = None):

    token = token or read_gh_token()

    m = re.match(r'^\s*(\d+)\s*(w|week|weeks|d|day|days|m|month|months|h|hour|hours|min|minute|minutes)\s*$', window or "", re.IGNORECASE)  # parser for window
    n = int(m.group(1))
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

    init_commit = fetch_commit(owner, repo, init_sha)
    format_date = (init_commit.get("commit") or {}).get("committer").get("date")
    start_day = format_date.split("T", 1)[0]
    end_day = (datetime.strptime(start_day, "%Y-%m-%d").date() + timedelta(days=days)).strftime("%Y-%m-%d")
    date_key = "committer-date"

    # finding the last commit taking into account the end_day boundary

    last = get_last_commit_in_window(owner, repo, start_day, end_day, token)

    if not last:
        return [], None, None


    # fetching attributes from the target commit

    target_sha = last["sha"]

    target = fetch_commit(owner, repo, target_sha)
    target_format_date = (target.get("commit")).get("committer").get("date")
    target_date = target_format_date.split("T", 1)[0]

    # fetching attributes from the initial commit

    init_full = fetch_commit(owner, repo, init_sha)
    init_format_date = (init_full.get("commit")).get("committer").get("date")
    init_date = init_format_date.split("T", 1)[0]

    added_lines = get_lines(owner, repo, init_sha, token)

    results: List[dict] = []
    mp: dict[str, Optional[str]] = {}

    for line in added_lines:
        if line.filename not in mp:
            mp[line.filename] = get_file_content(owner, repo, line.filename, ref=target_sha, token=token)
        curr_text = mp[line.filename]
        if curr_text is None:
            survived = "not survived"
        else:
            if line.content in curr_text.splitlines():
                survived = "survived"
            else:
                survived = "not survived"
        results.append({
            "repo": f"{owner}/{repo}",
            "init_sha": init_sha,
            "init_date": init_date,
            "target_sha": target_sha,
            "target_date": target_date,
            "filename": line.filename,
            "code": line.content,
            "survived": survived,
        })
    return results, target_sha, target_date

def compute_survival_entire(owner: str, repo: str, init_sha: str, window: str, token: Optional[str] = None):

    token = token or read_gh_token()
    m = re.match(r'^\s*(\d+)\s*(w|week|weeks|d|day|days|m|month|months|h|hour|hours|min|minute|minutes)\s*$', window or "", re.IGNORECASE)  # parser for window
    n = int(m.group(1))
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
    init_commit = fetch_commit(owner, repo, init_sha, token)
    if not init_commit:
        return None
    
    format_date = (init_commit.get("commit") or {}).get("committer").get("date")
    start_day = format_date.split("T", 1)[0]
    end_day = (datetime.strptime(start_day, "%Y-%m-%d").date() + timedelta(days=days)).strftime("%Y-%m-%d")
    date_key = "committer-date"

    # finding the last commit taking into account the end_day boundary
    added_lines = get_lines(owner, repo, init_sha, token)
    if not added_lines: 
        return None
    
    last = get_last_commit_in_window(owner, repo, start_day, end_day, token)

    if not last: 
        return None
    # fetching attributes from the initial commit

    init_full = fetch_commit(owner, repo, init_sha, token)
    init_format_date = (init_full.get("commit")).get("committer").get("date")
    init_date = init_format_date.split("T", 1)[0]


    # if not last:
    #     return ({
    #          "repo": f"{owner}/{repo}",
    #          "init_sha": init_sha,
    #          "init_date": init_date,
    #          "target_sha": None,
    #          "target_date": None,
    #          "survived_lines": len(added_lines),
    #          "added_lines_survival": len(added_lines),
    #          "survival_rate": 1,
    #      }, None, None)


    # fetching attributes from the target commit
    target_sha = last["sha"]
    target = fetch_commit(owner, repo, target_sha, token)
    if not target:
        return None
    target_format_date = (target.get("commit")).get("committer").get("date")
    target_date = target_format_date.split("T", 1)[0]

    results: List[dict] = []
    mp: dict[str, Optional[str]] = {}

    total = 0
    for line in added_lines:
        if line.filename not in mp:
            mp[line.filename] = get_file_content(owner, repo, line.filename, ref=target_sha, token=token)
        curr_text = mp[line.filename]
        if curr_text is None:
            survived = "not survived"
        else:
            if line.content in curr_text.splitlines():
                survived = "survived"
                total += 1
            else:
                survived = "not survived"
    rate = 0.00000

    if len(added_lines) != 0:
        rate = total / len(added_lines)
    result = {
        "repo": f"{owner}/{repo}",
        "init_sha": init_sha,
        "init_date": init_date,
        "target_sha": target_sha,
        "target_date": target_date,
        "survived_lines": total,
        "added_lines_survival": len(added_lines),
        "survival_rate": round(rate, 5),
    }
    return result, target_sha, target_date

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--owner", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--init_sha", required=True)
    p.add_argument("--window", required=True)
    args = p.parse_args()

    rows, target, target_date = compute_survival(
        owner=args.owner,
        repo=args.repo,
        init_sha=args.init_sha,
        window=args.window,
    )

    if not rows:
        rows = [{"repo": "", "init_sha": "", "init_date": "", "target_sha": "", "target_date": "",
                 "filename": "", "code": "", "survived": ""}]
    os.makedirs(os.path.dirname("saved_data/survival_test.csv"), exist_ok=True)
    pd.DataFrame(rows).to_csv("saved_data/survival_test.csv", index=False, encoding="utf-8")

    # Also mentioned the initial / target / diff commit for easier lookup

    base_commit_url = f"https://github.com/{args.owner}/{args.repo}/commit"
    base_compare_url = f"https://github.com/{args.owner}/{args.repo}/compare"
    init_url = f"{base_commit_url}/{args.init_sha}"

    # used for DEBUG and TESTING
    if target:
        target_url = f"{base_commit_url}/{target}"
        compare_url = f"{base_compare_url}/{args.init_sha}...{target}"
        print(f"Wrote to survival_test")
        print(f"Initial commit: {init_url}")
        print(f"Target commit:  {target_url}")
        print(f"Compare diff:   {compare_url}")
    else:
        print(f"No commits found!")
        print(f"Initial commit: {init_url}")

if __name__ == "__main__":
    _cli()

# python -m spoiler.scraping.survival --owner microsoft --repo vscode-pull-request-github --init_sha f4cb0b9452a508919f068a0fd0a0611ef1e35e42 --window 1w
