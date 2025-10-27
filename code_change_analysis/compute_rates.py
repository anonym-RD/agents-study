import os, json
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

import matplotlib.pyplot as plt

import pandas as pd
import requests
from datasets import load_dataset

import time
from survival import compute_survival_entire
from churn import compute_churn
from diffs import read_gh_token


agent_list = [
    ("Claude", "PullRequests_Claude", "Commits_ClaudeF", "Repositories_Claude"),
    ("Codex", "PullRequests_Codex", "Commits_CodexF", "Repositories_Codex"),
    ("Copilot", "PullRequests_Copilot", "Commits_CopilotF", "Repositories_Copilot"),
    ("Devin", "PullRequests_Devin", "Commits_DevinF", "Repositories_Devin"),
    ("Human", "PullRequests_Human", "Commits_HumanF", "Repositories_Human"),
    ("Jules", "PullRequests_Jules", "Commits_JulesF", "Repositories_Jules"),
]

@dataclass
class TokenInfo:
    token: str
    remaining: Optional[int] = None
    reset: Optional[int] = None


class TokenManager:

    def __init__(self, tokens: List[str]):
        self.tokens_info = [TokenInfo(token=t) for t in tokens]
        self.current_index = 0

    def get_token(self) -> str:
        token_info = self.tokens_info[self.current_index]
        now = time.time()
        if token_info.remaining == 0 and token_info.reset and now < token_info.reset:
            self.rotate_token()
        return self.tokens_info[self.current_index].token

    def rotate_token(self):
        for _ in range(len(self.tokens_info)):
            self.current_index = (self.current_index + 1) % len(self.tokens_info)
            token_info = self.tokens_info[self.current_index]
            now = time.time()
            if (token_info.remaining is None or token_info.remaining > 50) or (
                    token_info.reset and int(now) > token_info.reset):
                print(f"Switched to token index {self.current_index}")
                return

        rest = [(i,t.reset) for i, t in enumerate(self.tokens_info) if t.reset]
        if rest:
            ind, mini = min(rest, key=lambda t: t[1])
            wait = max(0, mini - int(time.time())) + 5
            print(f" All tokens used. Sleeping for {wait}s...")
            time.sleep(wait)

            for it in self.tokens_info:
                it.remaining = None
                it.reset = None
            self.current_index = ind
            print(f"Switched to token index {ind}")

    def update_limit(self, remaining: int, reset_timestamp: int):
        self.tokens_info[self.current_index].remaining = remaining
        self.tokens_info[self.current_index].reset = reset_timestamp

    def show_index(self) -> int:
        return self.current_index

    def get_reset_time(self) -> Optional[int]:
        return self.tokens_info[self.current_index].reset

    def get_remaining(self) -> Optional[int]:
        return self.tokens_info[self.current_index].remaining

    def get_all_reset_times(self) -> List[Optional[int]]:
        return [token_info.reset for token_info in self.tokens_info]

    def update_index(self, index: int):
        if 0 <= index < len(self.tokens_info):
            self.current_index = index
        else:
            raise IndexError("Token index out of range")

def check_github_rate_limit(token_manager: TokenManager):
    token = token_manager.get_token()
    #print(token)
    headers = {"Authorization": f"token {token}"} if token else {}
    url = "https://api.github.com/rate_limit"

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            return

        data = resp.json()
        core = data.get("resources", {}).get("core", {})
        remaining = core.get("remaining", 0)
        reset = core.get("reset", int(time.time()))
        token_manager.update_limit(remaining, reset)

    except Exception:
        return

def load_tokens():
    with open("github_tokens.json", "r") as f:
        data = json.load(f)
    return data.get("tokens", [])

def process_agent(agent_name, pr_config, commit_config, repo_config, window, limit, token, save_loc):
    print(f"Checking in PRs for {agent_name}\n")

    prs = load_dataset("dataset_hf", pr_config, split="train", cache_dir="/huggingfaceCache")
    repos = load_dataset("dataset_hf", repo_config, split="train", cache_dir="/huggingfaceCache")
    commits = load_dataset("dataset_hf", commit_config, split="train", cache_dir="/huggingfaceCache")
    num_total = limit          

    pan_prs = prs.to_pandas()
    pan_commits = commits.to_pandas()
    pan_repos = repos.to_pandas()
    pan_repos = pan_repos[pan_repos["role"] == "BASE"].copy()
    pan_prs_filtered = pan_prs[(pan_prs["state"].str.upper() == "MERGED")
        & (pan_prs["commits_count"].notnull())].copy()

    joined_table = pd.merge(
            pan_prs_filtered,
            pan_commits,
            left_on="id",
            right_on="pr_id",
            how="inner")

    joined_table = pd.merge(
        joined_table,
        pan_repos[["pr_id", "stargazer_count"]],
        on="pr_id",
        how="left",
    )

    joined_table["committed_date"] = pd.to_datetime(joined_table["committed_date"], errors="coerce")
    joined_table = joined_table.sort_values(by=["pr_id", "committed_date"], ascending=[True, True])
    if save_loc == 'copilot': 
        joined_table = joined_table.groupby("pr_id", as_index=False).nth(1)
    else:
        joined_table = joined_table.groupby("pr_id", as_index=False).first()
    joined_table["stargazer_count"] = joined_table["stargazer_count"].fillna(0)

    zero_star_prs = joined_table[joined_table["stargazer_count"] == 0].copy()
    non_zero_prs = joined_table[joined_table["stargazer_count"] > 0].copy()


    target_per_bin = num_total // 3  

    if len(non_zero_prs) < 2 * target_per_bin:
        num_non_zero_bins = 1
    else:
        num_non_zero_bins = 2


    if num_non_zero_bins == 1:
        min_star = non_zero_prs["stargazer_count"].min()
        max_star = non_zero_prs["stargazer_count"].max()
        non_zero_prs["star_bin"] = f"{int(min_star)}-{int(max_star)}"
    else:
        quantiles = non_zero_prs["stargazer_count"].quantile(np.linspace(0, 1, num_non_zero_bins + 1))
        unique_edges = np.unique(quantiles.values)
        print(unique_edges)
        # Fallback if duplicates cause collapse
        if len(unique_edges) <= 2:
            print("Warning: duplicate quantile edges detected — collapsing non-zero stars into a single bin.")
            min_star = non_zero_prs["stargazer_count"].min()
            max_star = non_zero_prs["stargazer_count"].max()
            non_zero_prs["star_bin"] = f"{int(min_star)}-{int(max_star)}"
            num_non_zero_bins = 1
        else:
            non_zero_prs["star_bin"] = pd.cut(
                non_zero_prs["stargazer_count"],
                bins=unique_edges,
                include_lowest=True,
                right=True,
                duplicates="drop"
            ).astype(str)


    zero_star_prs["star_bin"] = "0"
    
    all_bins = []
    
    if len(zero_star_prs) > 0:
        all_bins.append(("0", zero_star_prs.sample(frac=1, random_state=42).reset_index(drop=True)))
    for bin_label, group in non_zero_prs.groupby("star_bin"):
        all_bins.append((bin_label, group.sample(frac=1, random_state=42).reset_index(drop=True)))
    
    target_per_bin = num_total // len(all_bins)
    
    kon = 0
    results = []
    autosave_path = f"/{save_loc}/{save_loc}_rates_autosave.csv"

    for bin_label, df_bin in all_bins: 
        counter = 0
        for ind, pr in df_bin.iterrows():

            if counter >= target_per_bin or kon >= limit:
                break
            print(f"Processing PR {ind+1}\n")
            if (ind+1) % 100 == 0:
                print(f"We are running the commit #{ind+1}!")
            pr_id = pr.get("id_x")
            pr_url = pr.get("url_x")
            base_repo = pr.get("base_repository") or {}
            sha = pr.get("sha")

            base_repo_url = base_repo.get("url", "")

            if "github.com/" not in base_repo_url:
                continue

            if not pr_id or not base_repo_url or not sha: # missing diff info => skip
                continue

            try:
                owner, repo = base_repo_url.split("github.com/")[1].split("/", 1)
            except Exception:
                continue
                
            check_github_rate_limit(token)
            token_str = token.get_token()
            try:
                rows, target_sha, target_date = compute_survival_entire(
                    owner=owner,
                    repo=repo,
                    init_sha=sha,
                    window=window,
                    token=token_str
                )
            except Exception:
                continue

            rows = rows or {}
            survival_data = {
                "added_lines_survival": rows.get("added_lines_survival", 0),
                "survived_lines": rows.get("survived_lines", 0),
                "survival_rate": rows.get("survival_rate", 0.00000),
            }
            try:
                churn_data = compute_churn(
                    owner=owner,
                    repo=repo,
                    init_sha=sha,
                    window=window,
                    token=token_str
                )
            except Exception:
                continue
            churn_data = churn_data or {}
            diff_url = churn_data.get("diff_url")

            merged = {
                "agent": agent_name,
                "pr_id": pr_id,
                "pr_url": pr_url,
                "owner": owner,
                "repo": repo,
                "init_sha": sha,
                "target_sha": target_sha or (churn_data or {}).get("target_sha"),
                "diff_url": diff_url or (
                    f"https://github.com/{owner}/{repo}/compare/{sha}..."
                    f"{(target_sha or (churn_data or {}).get('target_sha') or '')}"
                ),
            }   # table merging

            merged.update(survival_data)
            merged["total_LOC"] = churn_data.get("total_LOC", 0)
            merged["added_lines"] = churn_data.get("added_lines", 0)
            merged["deleted_lines"] = churn_data.get("deleted_lines", 0)
            merged["churn_rate"] = churn_data.get("churn_rate", 0.00000)
            merged["delete_rate"] = churn_data.get("delete_rate", 0.00000)
            merged['overlap_files'] = churn_data.get('overlap_files', 0)

            results.append(merged)

            kon += 1
            counter += 1
            
            if kon % 10 == 0:
                pd.DataFrame(results).to_csv(autosave_path, index=False)
                print(f" Saved {kon} PRs")
                
        if kon >= limit: 
            break

    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--agent", type=int)
    p.add_argument("--save", type=str)
    p.add_argument("--start", type=int)
    p.add_argument("--end", type=int)
    args = p.parse_args()

    #os.makedirs("saved_data", exist_ok=True)

    tokens = load_tokens()
    if not tokens:
        tokens = [read_gh_token()]

    token_manager = TokenManager(tokens[args.start:args.end])

    list = []
    if args.agent is not None:
        agent_name, pr_config, commit_config, repo_config = agent_list[args.agent]
        list.extend(process_agent(agent_name, pr_config, commit_config, repo_config, args.window, args.limit, token_manager, args.save))
    else:
        for agent_name, pr_config, commit_config, repo_config in agent_list:
            res = process_agent(agent_name, pr_config, commit_config, repo_config, args.window, args.limit, token_manager, args.save)
            list.extend(res)

    if list:
        df = pd.DataFrame(list)
        save_path = f"/{args.save}/{args.save}_rates.csv"
        df.to_csv(save_path, index=False)
        print(f"\nSaved results to {args.save}")


if __name__ == "__main__":
    main()

# All commits: python -m spoiler.scraping.compute_rates --window 3w --limit 1
# Claude: python -m spoiler.scraping.compute_rates --window 3w --limit 1 --agent 0
# Codex: python -m spoiler.scraping.compute_rates --window 3w --limit 1 --agent 1
# Copilot: python -m spoiler.scraping.compute_rates --window 3w --limit 1 --agent 2
# Devin: python -m spoiler.scraping.compute_rates --window 3w --limit 1 --agent 3
# Human: python -m spoiler.scraping.compute_rates --window 3w --limit 1 --agent 4
# Jules: python -m spoiler.scraping.compute_rates --window 3w --limit 1 --agent 5
