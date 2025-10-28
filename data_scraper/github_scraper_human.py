import math
import os
from pathlib import Path
import random
import time
from typing import Iterator, List, Optional, Tuple, Union
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

    def request_with_backoff(self, filter: str, curr_start_date: str, end_date: str, curr_batch_size: int, cursor_after: str) -> dict:
        retries = 0
        backoff = 1
        batch_size = curr_batch_size
        query = self.build_pr_query(filter=filter, start_date=curr_start_date, end_date=end_date, first=batch_size, after=cursor_after)
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
                    if rate_limited:
                        retries = 0
                        backoff = 1
                        continue
                
                    print(f"GraphQL errors: {response['errors']}")  
                    return None  

                return response.get("data", {})
            except requests.exceptions.RequestException as e:
                if retries >= self.max_retires:
                    raise RuntimeError("Max retires exceeded. Changing batch size")
                
                retries += 1
                print(f"Retrying in {backoff} seconds due to error: {e}")
                time.sleep(backoff)
                backoff *= 2
    
    #TODO: Implement multithreading and split into 1-day ranges (if we plan to scrape large ranges of data)
    #TODO: Add functionality when there are more than 100 commits, reviews, comments, issues per PR (not likely but possible)
    def build_pr_query(self, filter:str, start_date:str, end_date:str, first: int, after: Optional[str]) -> str:
        query = f"""
            query {{
                search(type: ISSUE, query:"is:pr {filter} {self.pr_time_key}:{start_date}..{end_date} sort:created-asc", first: {first}, after: {f'"{after}"' if after else 'null'}) {{
                    issueCount
                    pageInfo {{
                        endCursor
                        hasNextPage
                    }}
                    nodes {{
                        ... on PullRequest {{
                            id
                            title
                            url
                            number
                            state
                            bodyText
                            createdAt
                            mergedAt
                            closedAt
                            updatedAt
                            lastEditedAt
                            publishedAt
                            reviewDecision
                            isDraft
                            changedFiles
                            isCrossRepository
                            locked
                            isInMergeQueue
                            additions
                            deletions
                            activeLockReason 
                            author {{
                                login
                                url
                                __typename 
                            }}
                            authorAssociation
                            labels(first: 5) {{
                                totalCount
                                nodes {{
                                    name
                                    description
                                }}
                            }}
                            timelineItems {{totalCount}}
                            assignees {{totalCount}}
                            closingIssuesReferences {{totalCount}}
                            commits {{totalCount}}
                            reviews {{totalCount}}
                            comments {{totalCount}}
                            files(first: 100) {{
                                totalCount
                                nodes {{
                                    additions
                                    deletions
                                    path 
                                    changeType
                                }}
                            }}
                            baseRefName
                            baseRefOid
                            baseRepository {{
                                id
                                name
                                nameWithOwner
                                description
                                url 
                                stargazerCount 
                                isFork
                                isArchived
                                isDisabled
                                isEmpty
                                isInOrganization
                                isLocked
                                isPrivate
                                isMirror
                                isTemplate
                                isUserConfigurationRepository 
                                forkCount
                                forkingAllowed
                                archivedAt
                                createdAt
                                pushedAt
                                updatedAt
                                sshUrl
                                visibility
                                lockReason
                                owner {{id, url, login, __typename}}
                                licenseInfo {{ name }}
                                defaultBranchRef {{ name }}
                                repositoryTopics(first: 5) {{
                                    totalCount
                                    nodes {{
                                        topic {{
                                            name
                                        }}
                                    }}
                                }}
                                primaryLanguage {{
                                    name
                                }}
                                languages(first: 5) {{
                                    totalCount
                                    nodes {{
                                        name
                                    }}
                                }}
                                watchers {{
                                    totalCount
                                }}
                            
                            }}
                            headRefName
                            headRefOid 
                            headRepository {{
                                id
                                name
                                nameWithOwner
                                description
                                url
                                stargazerCount 
                                isFork
                                isArchived
                                isDisabled
                                isEmpty
                                isInOrganization
                                isLocked
                                isPrivate
                                isMirror
                                isTemplate
                                isUserConfigurationRepository 
                                forkCount
                                forkingAllowed
                                archivedAt
                                createdAt
                                pushedAt
                                updatedAt
                                sshUrl
                                visibility
                                lockReason
                                owner {{id, url, login, __typename}}
                                licenseInfo {{ name }}
                                defaultBranchRef {{ name }}
                                repositoryTopics(first: 5) {{
                                    totalCount
                                    nodes {{
                                        topic {{
                                            name
                                        }}
                                    }}
                                }}
                                primaryLanguage {{
                                    name
                                }}
                                languages(first: 5) {{
                                    totalCount
                                    nodes {{
                                        name
                                    }}
                                }}
                                watchers {{
                                    totalCount
                                }}
                            }}
                        }}
                       
                    }}
                }}
            }}
            """
        return query
    
    
    def parse_author(self, author_entry: dict) -> Union[User, Bot, Mannequin, Organization, Enterprise, UserPeek]: 
        typename = author_entry['__typename'] or None 
        
        if not typename: 
            return None
        
        match typename: 
            case "User": 
                user_organizations = [OrganizationSummary(
                  id =  org['id'] or None, 
                  login=org['login'] or None, 
                  name=org['name'] or None, 
                  url=org['url'] or None 
                )
                for org in ((author_entry.get('organizations') or {}).get('nodes') or []) if org is not None
                ]
                
                return User(
                    id = author_entry['id'] or None,
                    # database_id=author_entry['databaseId'],
                    login=author_entry['login'] or None, 
                    url=author_entry['url'] or None,
                    typename=author_entry['__typename'] or None, 
                    name=author_entry['name'] or None,
                    email=author_entry['email'] or None,
                    bio=author_entry['bio'] or None,
                    company=author_entry['company'] or None, 
                    created_at=author_entry['createdAt'] or None, 
                    updated_at=author_entry['updatedAt'] or None, 
                    followers=(((author_entry.get('followers') or {}).get('totalCount')) or None), 
                    following=(((author_entry.get('following') or {}).get('totalCount')) or None), 
                    # enterprises=author_entry['enterprises']['totalCount'], 
                    is_employee=author_entry['isEmployee'] or None, 
                    is_hireable=author_entry['isHireable'] or None, 
                    location=author_entry['location'] or None,
                    organizations = user_organizations, 
                    organization_count = (((author_entry.get('organizations') or {}).get('totalCount')) or None), 
                    pull_requests = (((author_entry.get('pullRequests') or {}).get('totalCount')) or None), 
                    repositories= (((author_entry.get('repositories') or {}).get('totalCount')) or None), 
                    repositories_contributed_to=(((author_entry.get('repositoriesContributedTo') or {}).get('totalCount')) or None), 
                    commit_comments=(((author_entry.get('commitComments') or {}).get('totalCount')) or None), 
                    issues=(((author_entry.get('issues') or {}).get('totalCount')) or None), 
                    sponsors=author_entry['sponsors'] or None, 
                    sponsoring=author_entry['sponsoring'] or None, 
                    watching=author_entry['watching'] or None
                )  
            case "Mannequin": 
                return UserPeek(None, author_entry['login'] or None, author_entry['url'] or None, None, author_entry['__typename'] or None)
            case "Organization":
                return UserPeek(None, author_entry['login'] or None, author_entry['url'] or None, None, author_entry['__typename'] or None)
            case "EnterpriseUserAccount":
                return UserPeek(None, author_entry['login'] or None, author_entry['url'] or None, None, author_entry['__typename'] or None)
            
    def assign_intervals_over_days(self, span: int, days: int) -> List[Tuple[int, int]]:
        if span <= 0:
            raise ValueError("Span must be positive")
        
        intervals = []
        remaining = []
        
        intervals_in_a_cycle = 24 // span 
        
        full_cycles = days // intervals_in_a_cycle

        remainder = days - (full_cycles * intervals_in_a_cycle) 

        for i in range(days):
            start_hour = (i * span) % 24
            end_hour = (start_hour + span) % 24
            if i < days - remainder: 
                intervals.append((start_hour, end_hour))
            else:
                remaining.append((start_hour, end_hour)) 
                
        random.seed(42)
        all_intervals = intervals + remaining
        random.shuffle(all_intervals)
        return all_intervals           
             

    def scrape_prs(self, filter: str, start_date: str, end_date: str, total: int, interval_span:int) -> Iterator[PullRequest | Repository | None]:
            total_prs_yielded = 0
            start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            total_days = (end_dt - start_dt).days + 1 
            max_per_day = math.ceil(total / total_days)
            step = max(1, total_days * max_per_day // total) 
            sample_dates = [start_dt + timedelta(days=i) for i in range(0, total_days, step)]
            
            intervals = self.assign_intervals_over_days(interval_span, total_days)
            sample_dates_with_intervals = []
            
            for i, curr_dt in enumerate(sample_dates):
                interval = intervals[i % len(intervals)] 
                start_hour, end_hour = interval

                curr_start = curr_dt.replace(hour=start_hour, minute=0, second=0)
                curr_end = curr_dt.replace(hour=end_hour, minute=0, second=0)


                if end_hour == 0:
                    curr_end = curr_end.replace(hour=23, minute=59, second=59, microsecond=59)

                sample_dates_with_intervals.append((curr_start.strftime("%Y-%m-%dT%H:%M:%SZ"), curr_end.strftime("%Y-%m-%dT%H:%M:%SZ")))
            
            for i, curr_dt in enumerate(sample_dates_with_intervals):
                days_left = total_days - i
                remaining_target = total - total_prs_yielded
                day_limit = min(max_per_day, math.ceil(remaining_target / days_left))
                curr_start_date, curr_end_date = curr_dt
                print(curr_start_date)       
                cursor_after = None 
                curr_batch_size = self.batch_size
                batch_status = False 
                day_yielded = 0
                 
                while True: 
                    curr_batch_size = min(curr_batch_size, day_limit - day_yielded, total - total_prs_yielded)
                    print(curr_batch_size)
                    if curr_batch_size <= 0:
                        break
                    try: 
                        response = self.request_with_backoff(filter, curr_start_date, curr_end_date, curr_batch_size, cursor_after)
                        
                        data = response
                        print(data)
                        if not data:
                            print(f"No data found with backoff for {filter} between {start_date} and {end_date}")
                            return
                        
                        search_data = data.get("search")
                        entries = search_data.get("nodes")
                        
                        for entry in entries:
                            if entry is None:
                                print("Entry PR not found")
                                continue
                            
                            author_type = ((entry.get('author') or {}).get("__typename") or None)
                            
                            if not author_type or author_type == "Bot":
                                continue 
                            
                            pr_body = entry.get('bodyText') or None 
                            
                            phrases_to_exclude = [
                                "Generated with Claude Code",
                                "Co-Authored-By: Claude",
                                "About Codegen",
                                "Co-authored-by: openhands",
                                "Automatic fix generated by OpenHands"
                            ] 
                            
                            if pr_body == None: 
                                continue
                            pr_body = pr_body.lower()
                            
                            if any(p.lower() in pr_body for p in phrases_to_exclude):
                                continue

                            if total_prs_yielded >= total:
                                print(f"Reached total of {total} PRs, stopping.")
                                return
                            
                            pr_labels = [Label((label['name'] or None), (label['description'] or None)) for label in ((entry.get('labels') or {}).get("nodes") or []) if label is not None]
                            yield PullRequest(
                                id=entry['id'] or None, 
                                # database_id=entry['databaseId'], 
                                title = entry['title'] or None,
                                url = entry['url'] or None,
                                number = entry['number'] or None,
                                state = entry['state'] or None,
                                body = entry['bodyText'] or None,
                                created_at = entry['createdAt'] or None,
                                merged_at = entry['mergedAt'] or None,
                                closed_at = entry['closedAt'] or None, 
                                updated_at = entry['updatedAt'] or None, 
                                last_edited_at = entry['lastEditedAt'] or None, 
                                published_at = entry['publishedAt'] or None, 
                                review_decision = entry['reviewDecision'] or None, 
                                is_draft= entry['isDraft'] or None, 
                                changed_files=entry['changedFiles'] or None, 
                                is_cross_repository=entry['isCrossRepository'] or None, 
                                locked=entry['locked'] or None, 
                                is_in_merge_queue=entry['isInMergeQueue'] or None, 
                                additions=entry['additions'] or None, 
                                deletions=entry['deletions'] or None, 
                                active_lock_reason=entry['activeLockReason'] or None, 
                                # author=self.parse_author(entry['author']),
                                author = UserPeek(None, (((entry.get('author') or {}).get('login')) or None), (((entry.get('author') or {}).get('url')) or None), None, (((entry.get('author') or {}).get('__typename')) or None)),
                                author_association=entry['authorAssociation'] or None, 
                                label_count=(((entry.get('labels') or {}).get('totalCount')) or None), 
                                labels = pr_labels, 
                                timeline_count=(((entry.get('timelineItems') or {}).get('totalCount')) or None), 
                                # timeline_items=pr_timeline_items, 
                                closing_issues_count=(((entry.get('closingIssuesReferences') or {}).get('totalCount')) or None), 
                                assignees_count=(((entry.get('assignees') or {}).get('totalCount')) or None), 
                                comments_count=(((entry.get('comments') or {}).get('totalCount')) or None), 
                                reviews_count=(((entry.get('reviews') or {}).get('totalCount')) or None), 
                                commits_count=(((entry.get('commits') or {}).get('totalCount')) or None), 
                                files=[FileChange((file['additions'] or None), (file['deletions'] or None), (file['path'] or None), (file['changeType'] or None)) for file in ((entry.get("files") or {}).get("nodes") or []) if file is not None],
                                base_ref_name=entry['baseRefName'] or None,
                                base_ref_oid=entry['baseRefOid'] or None,
                                base_repository=RepositoryPeek((((entry.get('baseRepository') or {}).get('id')) or None), (((entry.get('baseRepository') or {}).get('name')) or None), (((entry.get('baseRepository') or {}).get('url')) or None)), 
                                head_ref_name=entry['headRefName'] or None, 
                                head_ref_oid=entry['headRefOid'] or None, 
                                head_repository=RepositoryPeek((((entry.get('headRepository') or {}).get('id')) or None), (((entry.get('headRepository') or {}).get('name')) or None), (((entry.get('headRepository') or {}).get('url')) or None)), 
                            )
                                        
                            if entry.get('baseRepository'):
                                repository_topics=[topic['topic']['name'] for topic in (((entry.get('baseRepository') or {}).get('repositoryTopics') or {}).get('nodes') or []) if topic is not None]
                                languages=[lang['name'] for lang in (((entry.get('baseRepository') or {}).get('languages') or {}).get('nodes') or []) if lang is not None]
                                base_repo = entry.get('baseRepository') or {}
                                primary_language = base_repo.get('primaryLanguage', {}).get('name') if base_repo.get('primaryLanguage') else None
                                license_info = base_repo.get('licenseInfo', {}).get('name') if base_repo.get('licenseInfo') else None

                                yield Repository(
                                    id=entry['baseRepository']['id'] or None,
                                    pr_id=entry['id'] or None,
                                    # database_id=entry['baseRepository']['databaseId'],
                                    role='BASE', 
                                    name=entry['baseRepository']['name'] or None,
                                    name_with_owner=entry['baseRepository']['nameWithOwner'] or None,
                                    description=entry['baseRepository']['description'] or None,
                                    url=entry['baseRepository']['url'] or None,
                                    stargazer_count=entry['baseRepository']['stargazerCount'] or None,
                                    is_fork=entry['baseRepository']['isFork'] or None,
                                    is_archived=entry['baseRepository']['isArchived'] or None,
                                    is_disabled=entry['baseRepository']['isDisabled'] or None,
                                    is_empty=entry['baseRepository']['isEmpty'] or None,
                                    is_in_organization=entry['baseRepository']['isInOrganization'] or None,
                                    is_locked=entry['baseRepository']['isLocked'] or None,
                                    is_private=entry['baseRepository']['isPrivate'] or None,
                                    is_mirror=entry['baseRepository']['isMirror'] or None,
                                    is_template=entry['baseRepository']['isTemplate'] or None,
                                    is_user_configuration_repository=entry['baseRepository']['isUserConfigurationRepository'] or None,
                                    fork_count=entry['baseRepository']['forkCount'] or None,
                                    forking_allowed=entry['baseRepository']['forkingAllowed'] or None,
                                    archived_at=entry['baseRepository']['archivedAt'] or None,
                                    created_at=entry['baseRepository']['createdAt'] or None,
                                    pushed_at=entry['baseRepository']['pushedAt'] or None,
                                    updated_at=entry['baseRepository']['updatedAt'] or None,
                                    ssh_url=entry['baseRepository']['sshUrl'] or None,
                                    visibility=entry['baseRepository']['visibility'] or None,
                                    lock_reason=entry['baseRepository']['lockReason'] or None,
                                    owner=UserPeek(((((entry.get('baseRepository') or {}).get('owner') or {}).get('id')) or None), ((((entry.get('baseRepository') or {}).get('owner') or {}).get('login')) or None), ((((entry.get('baseRepository') or {}).get('owner') or {}).get('url')) or None), None, ((((entry.get('baseRepository') or {}).get('owner') or {}).get('__typename')) or None)),
                                    license_info=license_info, 
                                    default_brach=((((entry.get('baseRepository') or {}).get('defaultBranchRef') or {}).get('name')) or None),
                                    repository_topics=repository_topics,
                                    topics_count=((((entry.get('baseRepository') or {}).get('repositoryTopics') or {}).get('totalCount')) or None),
                                    primary_language=primary_language,
                                    language_count=((((entry.get('baseRepository') or {}).get('languages') or {}).get('totalCount')) or None),
                                    languages=languages,
                                    watchers=((((entry.get('baseRepository') or {}).get('watchers') or {}).get('totalCount')) or None)
                                )
                                
                            if entry.get('headRepository'):
                                repository_topics=[topic['topic']['name'] for topic in (((entry.get('headRepository') or {}).get('repositoryTopics') or {}).get('nodes') or []) if topic is not None]
                                languages=[lang['name'] for lang in (((entry.get('headRepository') or {}).get('languages') or {}).get('nodes') or []) if lang is not None]
                                head_repo = entry.get('headRepository') or {}
                                primary_language = head_repo.get('primaryLanguage', {}).get('name') if head_repo.get('primaryLanguage') else None
                                license_info = head_repo.get('licenseInfo', {}).get('name') if head_repo.get('licenseInfo') else None
                             
                                yield Repository(
                                    id=entry['headRepository']['id'] or None,
                                    pr_id=entry['id'] or None,
                                    # database_id=entry['headRepository']['databaseId'],
                                    role='HEAD',
                                    name=entry['headRepository']['name'] or None,
                                    name_with_owner=entry['headRepository']['nameWithOwner'] or None,
                                    description=entry['headRepository']['description'] or None,
                                    url=entry['headRepository']['url'] or None,
                                    stargazer_count=entry['headRepository']['stargazerCount'] or None,
                                    is_fork=entry['headRepository']['isFork'] or None,
                                    is_archived=entry['headRepository']['isArchived'] or None,
                                    is_disabled=entry['headRepository']['isDisabled'] or None,
                                    is_empty=entry['headRepository']['isEmpty'] or None,
                                    is_in_organization=entry['headRepository']['isInOrganization'] or None,
                                    is_locked=entry['headRepository']['isLocked'] or None,
                                    is_private=entry['headRepository']['isPrivate'] or None,
                                    is_mirror=entry['headRepository']['isMirror'] or None,
                                    is_template=entry['headRepository']['isTemplate'] or None,
                                    is_user_configuration_repository=entry['headRepository']['isUserConfigurationRepository'] or None,
                                    fork_count=entry['headRepository']['forkCount'] or None,
                                    forking_allowed=entry['headRepository']['forkingAllowed'] or None,
                                    archived_at=entry['headRepository']['archivedAt'] or None,
                                    created_at=entry['headRepository']['createdAt'] or None,
                                    pushed_at=entry['headRepository']['pushedAt'] or None,
                                    updated_at=entry['headRepository']['updatedAt'] or None,
                                    ssh_url=entry['headRepository']['sshUrl'] or None,
                                    visibility=entry['headRepository']['visibility'] or None,
                                    lock_reason=entry['headRepository']['lockReason'] or None,
                                    owner=UserPeek(((((entry.get('headRepository') or {}).get('owner') or {}).get('id')) or None), ((((entry.get('headRepository') or {}).get('owner') or {}).get('login')) or None), ((((entry.get('headRepository') or {}).get('owner') or {}).get('url')) or None), None, ((((entry.get('headRepository') or {}).get('owner') or {}).get('__typename')) or None)),
                                    license_info=license_info, 
                                    default_brach=((((entry.get('headRepository') or {}).get('defaultBranchRef') or {}).get('name')) or None),
                                    repository_topics=repository_topics,
                                    topics_count=((((entry.get('headRepository') or {}).get('repositoryTopics') or {}).get('totalCount')) or None),
                                    primary_language=primary_language,
                                    language_count=((((entry.get('headRepository') or {}).get('languages') or {}).get('totalCount')) or None),
                                    languages=languages,
                                    watchers=((((entry.get('headRepository') or {}).get('watchers') or {}).get('totalCount')) or None)
                                )
                            
                            day_yielded += 1
                            total_prs_yielded += 1
                            
                            if total_prs_yielded >= total: 
                                print(f"Reached total of {total} PRs, stopping.")
                                return
                            
                            if day_yielded >= day_limit:
                                print(f"Total so far: {total_prs_yielded}")
                                break
                        
                        if day_yielded >= day_limit:
                            break    
                        
                        page_info = search_data.get("pageInfo") or {}

                        end_cursor = page_info.get("endCursor") or None
                        
                        if not end_cursor:
                            print(f"Total so far: {total_prs_yielded}")
                            break
                        else: 
                            cursor_after = end_cursor
                        
                        if curr_batch_size == 1: 
                            curr_batch_size = self.batch_size
                            print(f"Batch size reset to: {self.batch_size}")
                            batch_status = False

                        if batch_status: 
                            curr_batch_size = min(self.batch_size, curr_batch_size * 2)
                            print(f"Increasing batch size to {curr_batch_size}")
                            batch_status = False
                    
                    
                    except Exception as e:
                        if isinstance(e, RuntimeError):
                            print("Max retries reached, decreasing batch size.")
                            curr_batch_size = max(1, curr_batch_size // 2)
                            batch_status = True
                            print(f"New batch size: {curr_batch_size}")     
            
            
            print(f"Only found PRS: {total_prs_yielded}")
            return             
                    
                
                    
                         


    
        

    
