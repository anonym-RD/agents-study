import os
from pathlib import Path
import time
from typing import Iterator, List, Optional, Union
from huggingface_hub import login
import requests
from datetime import datetime, timedelta, timezone
from dtos import Committer, EnterpriseUserAccount, PrTest, UserPeek, PullRequest, Repository, PullRequestTest, User, Bot, Mannequin, Organization, Enterprise, OrganizationSummary, Domain, Enterprise, Label, FileChange, RepositoryPeek, Issue, Commit, IssueType, Review, Comment
from token_manager import TokenManager

class GitHubScraper:
     
    def __init__(self, tokens: List[Optional[str]] = None, graphql_url: str = "https://api.github.com/graphql", batch_size: int = 50, pr_time_key: str = "created"):
        self.tokens = tokens if tokens else [self.read_gh_token()]
        self.token_manager = TokenManager(self.tokens)
        self.graphql_url = graphql_url
        self.batch_size = batch_size
        self.pr_time_key = pr_time_key
        self.max_retires = 6
        self.token_limit = 0
        self.headers_graphql = self.build_header()
        
    def build_header(self) -> dict:
        self.token = self.token_manager.get_token()
        # print(f"Using token index: {self.token_manager.show_index()}, remaining: {self.token_manager.get_remaining()}, reset at: {datetime.fromtimestamp(self.token_manager.get_reset_time()).isoformat() if self.token_manager.get_reset_time() else 'N/A'}")
        # print(f"Using token index: {self.token_manager.show_index()}") 
        return {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json"}
    
    def read_gh_token(self) -> str: 
        token = os.getenv("GITHUB_TOKEN")  
        if token: 
            return token.strip()
        raise RuntimeError("Don't be lazy! Set ur token in the environment variable.")

    def get_rate_limit(self) -> dict:
        query = """
        query {
            rateLimit {
                limit
                cost
                remaining
                resetAt
            }
        }
        """
        header = self.build_header()
        response = requests.post(self.graphql_url, json={"query": query}, headers=header, timeout=10)
        response.raise_for_status()
        return response.json().get("data", {}).get("rateLimit", {})

    def request_with_backoff(self, pr_ids:str) -> dict:
        retries = 0
        backoff = 1
        query = self.build_pr_query(pr_ids=pr_ids)
        while True:
       
            try:
                headers = self.build_header()
                response = requests.post(self.graphql_url, json={"query": query}, headers=headers, timeout=30)
                response.raise_for_status()
                response = response.json()
                rate_limit = self.get_rate_limit()
                self.token_manager.update_limit(int(rate_limit.get("remaining", 0)), int(datetime.fromisoformat(rate_limit.get("resetAt").replace("Z", "+00:00")).timestamp()) if rate_limit.get("resetAt") else None)
                if "errors" in response:
                    rate_limited = False
                    for err in response["errors"]:
                        if err.get("type") in ("RATE_LIMITED", "RATE_LIMIT"):
                            rate_limited = True
                            print("Hit rate limit.")
                            rate_limit = self.get_rate_limit()
                            self.token_manager.update_limit(int(rate_limit.get("remaining", 0)), int(datetime.fromisoformat(rate_limit.get("resetAt").replace("Z", "+00:00")).timestamp()) if rate_limit.get("resetAt") else None)
                            try: 
                                self.token_manager.rotate_token()
                                time.sleep(2)
                            except RuntimeError:
                                print("All tokens exhausted, waiting for shortest token reset...")
                                reset_times = self.token_manager.get_all_reset_times()
                                print(reset_times)
                                now = time.time()
                                shortest_reset = min(reset_times)
                                print(shortest_reset) 
                                if shortest_reset < now: 
                                    print(f"Found one token with no wait time at index {reset_times.index(shortest_reset)}, switching to it.")
                                    self.token_manager.update_index(reset_times.index(shortest_reset)) 
                                else:
                                    self.token_manager.update_index(reset_times.index(shortest_reset)) 
                                    wait_time = shortest_reset - now + 100
                                    wait_time_minutes = wait_time / 60
                                    print(f"Waiting for {wait_time_minutes:.1f} minutes.")
                                    print(self.token_manager.show_index())
                                    time.sleep(wait_time)
                            break
                        elif err.get("type") == "NOT_FOUND":
                            print(f"PR not found. Continuing...")
                        else:
                            print(f"GraphQL errors: {response['errors']}")  
                            print(f"It is what it is, continuing...")
                            
                    if rate_limited:
                        retries = 0
                        backoff = 1
                        continue

                return response.get("data", {})
            except requests.exceptions.RequestException as e:
                if retries >= self.max_retires:
                   raise RuntimeError(f"Max retries reached.")
                retries += 1
                print(f"Retrying in {backoff} seconds due to error: {e}")
                time.sleep(backoff)
                backoff *= 2
    
    def build_pr_query(self, pr_ids: str) -> str:
        query = f"""
            query {{
                nodes(ids: [{pr_ids}]) {{
                    ... on PullRequest {{
                        id
                        closingIssuesReferences(first: 100) {{
                            totalCount
                            nodes {{
                                id
                                url
                                title
                                number
                                bodyText
                                author {{login, url, __typename}}
                                createdAt
                                lastEditedAt
                                publishedAt
                                updatedAt
                                locked
                                issueType {{
                                    name
                                    description
                                }}
                                closedByPullRequestsReferences {{totalCount}}
                                state
                                stateReason
                                trackedIssuesCount
                                labels(first: 5) {{
                                    totalCount
                                    nodes {{
                                        name
                                        description
                                    }}
                                }}
                            }}    
                        }}
                        comments(first: 100) {{
                                    totalCount
                                    nodes {{
                                        id
                                        url
                                        author {{ login, url, __typename }}
                                        bodyText
                                        createdAt
                                        lastEditedAt
                                        publishedAt
                                        updatedAt
                                        isMinimized
                                        minimizedReason   
                                    }}
                                }}
                        reviews(first: 100) {{
                            totalCount
                            nodes {{
                                id
                                url
                                author {{ login, url, __typename }}
                                bodyText
                                createdAt
                                updatedAt
                                lastEditedAt
                                publishedAt
                                submittedAt
                                isMinimized
                                minimizedReason
                                state
                            }}
                        }}
                        commits(first: 100) {{
                            totalCount 
                            nodes {{
                                commit {{
                                    oid
                                    id
                                    url
                                    commitUrl
                                    committedDate
                                    additions
                                    deletions
                                    authoredDate
                                    messageBody
                                    messageHeadline
                                    changedFilesIfAvailable
                                    authors(first: 2) {{
                                        totalCount
                                        nodes {{
                                            name
                                            email
                                        }}
                                    }}
                                    committer {{
                                        email
                                        name
                                    }}
                                }}
                            }}
                        }}     
                    }}
                }}
            }} 
    """
        return query
    
    def scrape_prs(self, pr_ids:list[str] ) -> Iterator[Comment | Review | Issue | Commit | None]:
             curr_batch_size = self.batch_size
             batch_status = False 
             i = 0
             while i < len(pr_ids):
                id_batch = pr_ids[i:i+curr_batch_size]
                id_batch_str = ",".join(f'"{id_}"' for id_ in id_batch)
                try: 
                    response = self.request_with_backoff(id_batch_str)
                    data = response
                    
                    if not data:
                        print(f"No data found for PRs")
                        # Scoate return urile astea cand devine batch ul foarte mic pt cursor
                        return
                    
                    initial_entries = data.get("nodes") or []
                    entries = [entry for entry in initial_entries if entry is not None]
                    
                    if not entries:
                        print(f"No data found for PRs")
                        # Scoate return urile astea cand devine batch ul foarte mic pt cursor
                        return 
                    
                    for entry in entries:
                        
                        issues = (entry.get('closingIssuesReferences') or {}).get('nodes') or []
                        
                        if issues:
                            for issue in issues:
                                if issue is None or issue.get('author') is None:
                                    print("Skipping issue due to missing data.")
                                    continue
                                author = issue.get('author') or {}
                                author_peek = None
                                if author:
                                    author_peek = UserPeek(
                                        None,
                                        author.get('login') or None,
                                        author.get('url') or None,
                                        None,
                                        author.get('__typename') or None
                                    )
                                issue_type_field = issue.get('issueType') or None
                                issue_type_obj = None
                                if issue_type_field:
                                    issue_type_obj = IssueType(
                                        issue_type_field.get('name') or None,
                                        issue_type_field.get('description') or None
                                    )
                                # closed_by_prs = [pr_id['id'] for pr_id in issue.get('closedByPullRequestsReferences', {}).get('nodes', []) if pr_id is not None]
                                labels = [Label((label['name'] or None), (label['description'] or None)) for label in ((issue.get('labels') or {}).get("nodes") or []) if label is not None]
                                
                                yield Issue(
                                    id=issue['id'] or None,
                                    pr_id = entry['id'] or None,
                                    # database_id=issue['databaseId'],
                                    url=issue['url'] or None,
                                    title=issue['title'] or None,
                                    body=issue['bodyText'] or None,
                                    author=author_peek,
                                    created_at=issue['createdAt'] or None,
                                    last_edited_at=issue['lastEditedAt'] or None,
                                    published_at=issue['publishedAt'] or None,
                                    updated_at=issue['updatedAt'] or None,
                                    locked=issue['locked'] or None,
                                    issue_type=issue_type_obj,
                                    prs_closing_issue=(((issue.get('closedByPullRequestsReferences') or {}).get('totalCount')) or None),
                                    # pr_ids=closed_by_prs, 
                                    number=issue['number'] or None,
                                    state=issue['state'] or None,
                                    state_reason=issue['stateReason'] or None,
                                    # sub_issues_total=issue['subIssues']['totalCount'],
                                    tracked_issues_count=issue['trackedIssuesCount'] or None,
                                    labels=labels,
                                    label_count=(((issue.get('labels') or {}).get('totalCount')) or None)
                                )
                                
                        comments = (entry.get('comments') or {}).get('nodes') or []    
                        if comments: 
                            for comment in comments:
                                if comment is None or comment.get('author') is None:
                                    print("Skipping comment due to missing data.")
                                    continue
                                author = comment.get('author') or {}
                                author_peek = None
                                if author:
                                    author_peek = UserPeek(
                                        None,
                                        author.get('login') or None,
                                        author.get('url') or None,
                                        None,
                                        author.get('__typename') or None
                                    )
                                yield Comment(
                                    id=comment['id'] or None,
                                    # database_id=comment['databaseId'],
                                    pr_id=entry['id'] or None,
                                    url=comment['url'] or None,
                                    author=author_peek,
                                    body=comment['bodyText'] or None,
                                    created_at=comment['createdAt'] or None,
                                    last_edited_at=comment['lastEditedAt'] or None,
                                    published_at=comment['publishedAt'] or None,
                                    updated_at=comment['updatedAt'] or None,
                                    is_minimized=comment['isMinimized'] or None,
                                    minimized_reason=comment['minimizedReason'] or None
                                )
                        reviews = (entry.get('reviews') or {}).get('nodes') or []        
                                
                        if reviews:
                            for review in reviews:
                                if review is None or review.get('author') is None:
                                    print("Skipping review due to missing data.")
                                    continue
                                author = review.get('author') or {}
                                author_peek = None
                                if author:
                                    author_peek = UserPeek(
                                        None,
                                        author.get('login') or None,
                                        author.get('url') or None,
                                        None,
                                        author.get('__typename') or None
                                    )
                                yield Review(
                                    id=review['id'] or None,
                                    # full_database_id=review['fullDatabaseId'],
                                    pr_id=entry['id'] or None,
                                    url=review['url'] or None,
                                    author=author_peek,
                                    body=review['bodyText'] or None,
                                    created_at=review['createdAt'] or None,
                                    updated_at=review['updatedAt'] or None,
                                    last_edited_at=review['lastEditedAt'] or None,
                                    published_at=review['publishedAt'] or None,
                                    submitted_at=review['submittedAt'] or None,
                                    is_minimized=review['isMinimized'] or None,
                                    minimized_reason=review['minimizedReason'] or None,
                                    state=review['state'] or None
                                )
                        commits = (entry.get('commits') or {}).get('nodes') or []        
                        if commits:
                            for commit in commits or []:
                                if commit is None or commit.get('commit') is None:
                                    print("Skipping commit due to missing data.")
                                    continue
                                committer_info = None
                                changed_files = (commit.get('commit') or {}).get('changedFilesIfAvailable') or None
                                commit_authors = ((commit.get('commit') or {}).get('authors') or {}).get('nodes') or []
                                committer = (commit.get('commit') or {}).get('committer') or {}
                                if commit_authors:
                                    commit_authors = [Committer(committer['name'], committer['email']) for committer in commit_authors]
                                if committer: 
                                    committer_info = Committer(committer['name'], committer['email'])
                                    
                                yield Commit(
                                    id=commit['commit']['id'] or None,
                                    sha=commit['commit']['oid'] or None,
                                    pr_id=entry['id'] or None,
                                    url=commit['commit']['url'] or None,
                                    # commit_url=commit['commit']['commitUrl'],
                                    committed_date=commit['commit']['committedDate'] or None,
                                    additions=commit['commit']['additions'] or None,
                                    deletions=commit['commit']['deletions'] or None,
                                    authored_date=commit['commit']['authoredDate'] or None,
                                    message_body=commit['commit']['messageBody'] or None,
                                    message_headline=commit['commit']['messageHeadline'] or None,
                                    changed_files=changed_files,
                                    authors = commit_authors,
                                    author_count=((((commit.get('commit') or {}).get('authors') or {}).get('totalCount')) or None), 
                                    committer=committer_info
                                )
                    print(f"Yielded {i + len(entries)} PRs so far...")                
                    i += curr_batch_size
                    
                    if batch_status: 
                        curr_batch_size = min(self.batch_size, curr_batch_size * 2)
                        print(f"Increasing batch size to {curr_batch_size}")
                        batch_status = False

                except RuntimeError as e: 
                    print("Max retries reached, decreasing batch size.")
                    curr_batch_size = max(1, curr_batch_size // 2)
                    batch_status = True
                    print(f"New batch size: {curr_batch_size}")


  