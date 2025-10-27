import os
from pathlib import Path
import time
from typing import Iterator, List, Optional, Union
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
                    # if batch_size - 5 <= 0: 
                    #     print("Batch is 0")
                    #     return
                    # batch_size -= 5
                    # query = self.build_pr_query(filter=filter, start_date=curr_start_date, end_date=end_date, first=batch_size, after=cursor_after)
                    # print("Changed batch size")
                    # retries = 0
                    # backoff = 1
                    # continue
                
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
                                
                                ... on User {{
                                    id
                                    name
                                    email
                                    bio
                                    company 
                                    createdAt
                                    updatedAt
                                    followers {{
                                        totalCount
                                    }}
                                    following {{
                                        totalCount
                                    }} 
                                    isEmployee 
                                    isHireable 
                                    location
                                    organizations(first: 3) {{
                                        totalCount
                                        nodes {{
                                            id
                                            login
                                            name
                                            url
                                        }}
                                    }}
                                    pullRequests {{
                                        totalCount
                                    }}
                                    repositories {{
                                        totalCount
                                    }}
                                    repositoriesContributedTo {{
                                        totalCount
                                    }}
                                    commitComments {{
                                        totalCount
                                    }}
                                    issues {{
                                        totalCount
                                    }}
                                    sponsors {{
                                        totalCount
                                    }}
                                    sponsoring {{
                                        totalCount
                                    }}
                                    watching {{
                                        totalCount
                                    }}
                                }}
                                
                                ... on Bot {{
                                    id 
                                    createdAt
                                    updatedAt
                                    
                                }}
                                
                                ... on Mannequin {{
                                    id 
                                    name 
                                    email 
                                    createdAt
                                    updatedAt
                                    claimant {{
                                        id
                                        __typename
                                        login
                                        name
                                        url
                                    }}  
                                }}
                                
                                ... on Organization {{
                                    id 
                                    name
                                    email
                                    description
                                    createdAt
                                    updatedAt
                                    archivedAt
                                    isVerified
                                    location
                                    mannequins {{
                                        totalCount
                                    }}
                                    domains(first: 3) {{
                                        totalCount
                                        nodes {{
                                            id 
                                            domain
                                        }}
                                    }}
                                    enterpriseOwners(first: 3) {{
                                        totalCount 
                                        nodes {{
                                            id
                                            __typename 
                                            login
                                            name
                                            url
                                        }}
                                    }}
                                    repositories {{
                                        totalCount
                                    }}
                                    sponsors {{
                                        totalCount
                                    }}
                                    sponsoring {{
                                        totalCount
                                    }}
                                    teams {{
                                        totalCount
                                    }} 
                                }}
                                
                                ... on EnterpriseUserAccount {{
                                    id 
                                    name
                                    createdAt
                                    updatedAt
                                    organizations(first: 3) {{
                                        totalCount
                                        nodes {{
                                            id
                                            login
                                            name
                                            url
                                        }}
                                    }}
                                    enterprise {{
                                        id
                                        name 
                                        description
                                        location
                                        members {{
                                            totalCount
                                        }}
                                        organizations {{
                                            totalCount
                                        }}
                                        
                                    }}
                                }}
                            
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
    
    
    def parse_author(self, author_entry: dict) -> Union[User, Bot, Mannequin, Organization, Enterprise]: 
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
            case "Bot":
                return Bot(
                    id = author_entry['id'] or None, 
                    # database_id=author_entry['databaseId'], 
                    login=author_entry['login'] or None, 
                    url=author_entry['url'] or None, 
                    typename=author_entry['__typename'] or None,
                    created_at=author_entry['createdAt'] or None, 
                    updated_at=author_entry['updatedAt'] or None
                )
            case "Mannequin": 
                return Mannequin(
                    id = author_entry['id'] or None, 
                    # database_id=author_entry['databaseId'], 
                    login=author_entry['login'] or None, 
                    url=author_entry['url'] or None,
                    name=author_entry['name'] or None,
                    email=author_entry['email'] or None, 
                    typename=author_entry['__typename'] or None,
                    created_at=author_entry['createdAt'] or None, 
                    updated_at=author_entry['updatedAt'] or None,
                    claimant=UserPeek((((author_entry.get('claimant') or {}).get('id')) or None), (((author_entry.get('claimant') or {}).get('login')) or None), (((author_entry.get('claimant') or {}).get('url')) or None), (((author_entry.get('claimant') or {}).get('name')) or None), (((author_entry.get('claimant') or {}).get('__typename')) or None))
                ) 
            case "Organization":
                organization_domains = [
                    Domain(
                        id = domain["id"] or None, 
                        domain = domain['domain'] or None
                    )
                    for domain in ((author_entry.get("domains") or {}).get("nodes") or []) if domain is not None
                ]
                
                organization_enterprise_owners = [
                    UserPeek(
                        id = owner['id'] or None, 
                        login = owner['login'] or None, 
                        url = owner['url'] or None, 
                        name = owner['name'] or None, 
                        typename=owner['__typename'] or None
                    )
                    for owner in ((author_entry.get('enterpriseOwners') or {}).get("nodes") or []) if owner is not None
                ]
                 
                return Organization(
                    id = author_entry['id'] or None, 
                    # database_id=author_entry['databaseId'], 
                    login=author_entry['login'] or None, 
                    url=author_entry['url'] or None,
                    name=author_entry['name'] or None,
                    email=author_entry['email'] or None,
                    description=author_entry['description'] or None,  
                    typename=author_entry['__typename'] or None,
                    created_at=author_entry['createdAt'] or None, 
                    updated_at=author_entry['updatedAt'] or None,
                    archived_at=author_entry['archivedAt'] or None,
                    is_verified=author_entry['isVerified'] or None, 
                    location=author_entry['location'] or None, 
                    mannequins=(((author_entry.get('mannequins') or {}).get('totalCount')) or None),
                    domains=organization_domains, 
                    domain_count=(((author_entry.get('domains') or {}).get('totalCount')) or None), 
                    enterprise_owners=organization_enterprise_owners, 
                    enterprise_owners_count=(((author_entry.get('enterpriseOwners') or {}).get('totalCount')) or None), 
                    repositories=(((author_entry.get('repositories') or {}).get('totalCount')) or None), 
                    sponsors=(((author_entry.get('sponsors') or {}).get('totalCount')) or None), 
                    sponsoring=(((author_entry.get('sponsoring') or {}).get('totalCount')) or None), 
                    teams=(((author_entry.get('teams') or {}).get('totalCount')) or None)    
                )
            case "EnterpriseUserAccount":
                enterprise_organizations = [
                    OrganizationSummary(
                        id = org['id'] or None, 
                        login = org['login'] or None, 
                        name = org['name'] or None, 
                        url = org['url'] or None
                    )
                    for org in ((author_entry.get('organizations') or {}).get("nodes") or []) if org is not None
                ]
                
                return EnterpriseUserAccount(
                    id = author_entry['id'] or None, 
                    login=author_entry['login'] or None, 
                    url=author_entry['url'] or None,
                    name=author_entry['name'] or None,
                    typename=author_entry['__typename'] or None,
                    created_at=author_entry['createdAt'] or None, 
                    updated_at=author_entry['updatedAt'] or None,
                    organizations=enterprise_organizations, 
                    organization_count=(((author_entry.get('organizations') or {}).get('totalCount')) or None), 
                    enterprise=Enterprise((((author_entry.get('enterprise') or {}).get('id')) or None), (((author_entry.get('enterprise') or {}).get('name')) or None), (((author_entry.get('enterprise') or {}).get('description')) or None), (((author_entry.get('enterprise') or {}).get('location')) or None), ((((author_entry.get('enterprise') or {}).get('members') or {}).get('totalCount')) or None), ((((author_entry.get('enterprise') or {}).get('organizations') or {}).get('totalCount')) or None)) 
                )
                        

    def scrape_prs(self, filter: str, start_date: str, end_date: str, total: int) -> Iterator[PullRequest | Repository | None]:
             curr_start_date = start_date 
             cursor_after = None # for pagination
             num_res_page = 0 # results returned on the current page
             curr_batch_size = self.batch_size 
             total_prs_yielded = 0
             batch_status = False
             while True: 

                try: 
                    response = self.request_with_backoff(filter, curr_start_date, end_date, curr_batch_size, cursor_after)
                    
                    data = response
                    
                    if not data:
                        print(f"No data found with backoff for {filter} between {start_date} and {end_date}")
                        return
                        
                    if "search" not in data or len((data["search"].get("nodes") or [])) == 0:
                        #TODO: This can be commented out and return directly (is different than REST pagination)
                        # if curr_start_date == end_date:
                        #     return
                        print(f"No data found from {curr_start_date}. Continuing ...")
                        # return
                        curr_start_dt = datetime.fromisoformat(curr_start_date.replace("Z", "+00:00"))
                        end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                        next_start_dt = min(curr_start_dt + timedelta(days=1), end_dt)
                        curr_start_date = next_start_dt.isoformat().replace("+00:00", "Z")
                        cursor_after = None
                        num_res_page = 0
                        continue
                    
                    search_data = data.get("search")
                    entries = search_data.get("nodes")
                    #print(f"Found {search_data['issueCount']} entries between {curr_start_date} and {end_date}")
                    if total_prs_yielded % 100 == 0 and total_prs_yielded > 0:   
                        print(f"Found {total_prs_yielded} PRs so far, yielding more ...")
                    
                    for entry in entries:
                        if entry is None:
                            num_res_page += 1
                            continue
                        
                        if total_prs_yielded == total:
                            print(f"Reached total of {total} PRs, stopping.")
                            return 

                        pr_labels = [Label((label['name'] or None), (label['description'] or None)) for label in ((entry.get('labels') or {}).get("nodes") or []) if label is not None]
                        # pr_timeline_items = [typename for typename in entry.get('timelineItems', {}).get('nodes', []) if typename is not None]
                       
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
                            author=self.parse_author(entry['author']),
                            # author = UserPeek(None, (((entry.get('author') or {}).get('login')) or None), (((entry.get('author') or {}).get('url')) or None), None, (((entry.get('author') or {}).get('__typename')) or None)),
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
                            # primary_language = entry.get('baseRepository', {}).get('primaryLanguage', {}).get('name') if entry.get('baseRepository', {}).get('primaryLanguage') else None
                            # license_info = entry.get('baseRepository', {}).get('licenseInfo', {}).get('name') if entry.get('baseRepository', {}).get('licenseInfo') else None
                            yield Repository(
                                id=entry['baseRepository']['id'] or None,
                                pr_id=entry['id'] or None,
                                # database_id=entry['baseRepository']['databaseId'],
                                role='BASE', # enums would work here, but I didn t bother modifying the dictionary conversion method
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
                            # primary_language = entry.get('headRepository', {}).get('primaryLanguage', {}).get('name') if entry.get('headRepository', {}).get('primaryLanguage') else None
                            # license_info = entry.get('headRepository', {}).get('licenseInfo', {}).get('name') if entry.get('headRepository', {}).get('licenseInfo') else None
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
                        
                        total_prs_yielded += 1

                    
                    if total_prs_yielded == search_data['issueCount']:
                        print(f"All {total_prs_yielded} PRs have been yielded between {start_date} and {end_date}.")
                        return
                    
                    page_info = search_data.get("pageInfo") or {}
                    #has_next_page = page_info.get("hasNextPage", False)
                    end_cursor = page_info.get("endCursor") or None

                    
                    if num_res_page + 2 * curr_batch_size < 1000: 
                        cursor_after = end_cursor
                        num_res_page += curr_batch_size
                    elif curr_start_date == entries[-1]["createdAt"]:
                        print("ENTERED WITH SAME START TIME") 
                        # TODO: We can be even more granular and move just a second and keep the cursor if any
                        curr_start_dt = datetime.fromisoformat(curr_start_date.replace("Z", "+00:00"))
                        end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                        next_start_dt = min(curr_start_dt + timedelta(days=1), end_dt)
                        curr_start_date = next_start_dt.isoformat().replace("+00:00", "Z")
                        cursor_after = None
                        num_res_page = 0
                    else: 
                        print("ENTERED")
                        curr_start_date = entries[-1]["createdAt"]
                        cursor_after = None
                        num_res_page = 0
                    
                    if search_data["issueCount"] <= self.batch_size:
                        print("All PRs found!")
                        return

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
                    print(e)
                    
                
                    
                         


    
        

    
