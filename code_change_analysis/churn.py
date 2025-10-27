from __future__ import annotations
from typing import List, Optional, Dict
import subprocess
import argparse
import os
import re
from datetime import datetime, timedelta
import pandas as pd
import requests

from diffs import read_gh_token, fetch_commit, fetch_compare_diffs, get_last_commit_in_window


def compute_churn(owner: str, repo: str, init_sha: str, window: str, token: Optional[str] = None):

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
    
    init_files = set(f["filename"] for f in init_commit.get("files", []) if "filename" in f)
    if not init_files:
        return None
    
    format_date = (init_commit.get("commit") or {}).get("committer").get("date")
    start_day = format_date.split("T", 1)[0]
    end_day = (datetime.strptime(start_day, "%Y-%m-%d").date() + timedelta(days=days)).strftime("%Y-%m-%d")
    date_key = "committer-date"

    # finding the last commit taking into account the end_day boundary

    last = get_last_commit_in_window(owner, repo, start_day, end_day, token)

    if not last:
        return None

    target_sha = last["sha"]
    target_commit = fetch_commit(owner, repo, target_sha, token)
    if not target_commit:
        return None
    target_date = target_commit["commit"]["committer"]["date"].split("T", 1)[0]


    diff = fetch_compare_diffs(owner, repo, init_sha, target_sha, token)
    if not diff:
        return None
    
    filtered_files = [f for f in diff.get("files", []) if f.get("filename") in init_files]
    
    n_same_files = len(filtered_files)
    
    if not filtered_files: 
        return {
            "repo": f"{owner}/{repo}",
            "init_sha": init_sha,
            "init_date": start_day,
            "target_sha": target_sha,
            "target_date": target_date,
            "overlap_files": n_same_files,
            "total_LOC": 0,
            "added_lines": 0,
            "deleted_lines": 0,
            "churn_rate": 0.0,
            "delete_rate": 0.0,
            "diff_url": f"https://github.com/{owner}/{repo}/compare/{init_sha}...{target_sha}",
        }

    total_additions = 0
    total_deletions = 0
    total_loc = 0

    for file in filtered_files:
        edits = file.get("edits", [])
        total_loc += len(edits)
        for i in range(len(edits)):
            if edits[i].get("type") == "del":
                if i + 1 < len(edits) and edits[i + 1].get("type") == "add":
                    continue  # modification

                total_deletions += 1

            elif edits[i].get("type") == "add":
                total_additions += 1


    if total_loc == 0:
        churn_rate = 0.0
        delete_rate = 0.0
    else:
        churn_rate = total_additions / total_loc
        delete_rate = total_deletions / total_loc

    return {
        "repo": f"{owner}/{repo}",
        "init_sha": init_sha,
        "init_date": start_day,
        "target_sha": target_sha,
        "target_date": target_date,
        "overlap_files": n_same_files,
        "total_LOC": total_loc,
        "added_lines": total_additions,
        "deleted_lines": total_deletions,
        "churn_rate": round(churn_rate, 5),
        "delete_rate": round(delete_rate, 5),
        "diff_url": f"https://github.com/{owner}/{repo}/compare/{init_sha}...{target_sha}",
    }


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--owner", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--init_sha", required=True)
    p.add_argument("--window", required=True)
    args = p.parse_args()

    result = compute_churn(
        owner=args.owner,
        repo=args.repo,
        init_sha=args.init_sha,
        window=args.window,
    )

    # commit_url = f"https://github.com/{args.owner}/{args.repo}/commit/{args.init_sha}"
    # print(f"Init commit: {commit_url}\n")

    os.makedirs(os.path.dirname("saved_data/churn_test.csv"), exist_ok=True)
    pd.DataFrame([result]).to_csv("saved_data/churn_test.csv", index=False, encoding="utf-8")
    print("Wrote to churn_test.csv")


if __name__ == "__main__":
    _cli()

# python -m spoiler.scraping.churn --owner microsoft --repo vscode-pull-request-github --init_sha f4cb0b9452a508919f068a0fd0a0611ef1e35e42 --window 1w
# python -m spoiler.scraping.churn --owner microsoft --repo vscode-pull-request-github --init_sha c281f31e5e0b2f1e5e6c872e1c80a1861e8e09ec --window 1w