from dataclasses import dataclass, asdict
from typing import Optional, List, Union
import json

class Serializable:
    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class OrganizationSummary(Serializable):
    id: str
    login: str
    url: str
    name: Optional[str] = None

@dataclass
class Domain(Serializable): 
    id: str 
    domain: str 
    
@dataclass
class UserPeek(Serializable):
    id: Optional[str] = None
    login: Optional[str]  = None
    url: Optional[str]  = None
    name: Optional[str] = None
    typename: Optional[str]  = None
    
@dataclass
class User(Serializable): 
    id: str 
    # database_id: int 
    login: str 
    name: str
    email: str  
    url: str 
    typename: str 
    created_at: str 
    is_employee: bool 
    is_hireable: bool
    followers: int 
    following: int 
    # enterprises: int 
    organization_count: int  
    pull_requests: int 
    repositories: int 
    repositories_contributed_to: int 
    commit_comments: int 
    issues: int 
    sponsors: int 
    sponsoring: int 
    watching: int
    bio: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    updated_at: Optional[str] = None
    organizations: Optional[List[OrganizationSummary]] = None 
    
@dataclass
class Bot(Serializable): 
    id: str 
    # database_id: int 
    login: str 
    url: str 
    typename: str 
    created_at: str 
    updated_at: Optional[str] = None

@dataclass
class Mannequin(Serializable): 
    id: str 
    # database_id: int 
    login: str 
    url: str 
    typename: str 
    name: str 
    email: str
    created_at: str 
    updated_at: Optional[str] = None
    claimant: Optional[UserPeek] = None 
    
@dataclass
class Organization(Serializable): 
    id: str 
    # database_id: int 
    login: str 
    url: str 
    typename: str 
    name: str 
    email: str
    created_at: str 
    is_verified: bool 
    mannequins: int 
    domain_count: int 
    enterprise_owners_count: int 
    repositories: int 
    sponsors: int 
    sponsoring: int 
    teams: int
    updated_at: Optional[str] = None
    archived_at: Optional[str] = None 
    description: Optional[str] = None 
    enterprise_owners: Optional[List[UserPeek]] = None 
    domains: Optional[List[Domain]] = None
    location: Optional[str] = None 
    
@dataclass
class Enterprise(Serializable): 
    id: str 
    name: str 
    members: int 
    organizations: int 
    description: Optional[str] = None 
    location: Optional[str] = None 

@dataclass
class EnterpriseUserAccount(Serializable): 
    id: str 
    login: str 
    url: str 
    typename: str 
    name: str 
    created_at: str 
    organization_count: int  
    enterprise: Optional[Enterprise] = None 
    updated_at: Optional[str] = None
    organizations: Optional[List[OrganizationSummary]] = None
    
@dataclass
class Label(Serializable): 
    name: str 
    description: Optional[str]  = None
    
@dataclass
class IssueType(Serializable): 
    name: str 
    description: Optional[str]  = None

@dataclass
class Issue(Serializable): 
    id: str 
    # database_id: int
    pr_id: str
    url: str 
    title: str 
    body: str 
    created_at: str 
    locked: bool 
    number: int 
    state: str 
    # sub_issues_total: int 
    tracked_issues_count: int 
    label_count: int 
    last_edited_at: Optional[str] = None
    published_at: Optional[str] = None
    updated_at: Optional[str] = None
    issue_type: Optional[IssueType] = None
    labels: Optional[List[Label]] = None
    state_reason: Optional[str] = None
    author: Optional[UserPeek] = None
    pr_ids: Optional[List[str]] = None
    prs_closing_issue: Optional[int] = None 
    
@dataclass
class Comment(Serializable): 
    id: str 
    # database_id: int 
    pr_id: str
    url: str 
    body: str 
    created_at: str 
    is_minimized: bool 
    minimized_reason: Optional[str] = None
    last_edited_at: Optional[str] = None
    published_at: Optional[str] = None
    updated_at: Optional[str] = None
    author: Optional[UserPeek] = None
    
@dataclass
class Review(Serializable): 
    id: str 
    # full_database_id: int
    pr_id: str 
    url: str 
    body: str 
    created_at: str 
    is_minimized: bool 
    state: str 
    updated_at: Optional[str] = None
    last_edited_at: Optional[str] = None
    published_at: Optional[str] = None
    submitted_at: Optional[str] = None
    minimized_reason: Optional[str] = None
    author: Optional[UserPeek] = None
    
@dataclass
class Committer(Serializable): 
    name: str 
    email: str 

@dataclass
class Commit(Serializable):  
    id: str 
    sha: str
    pr_id: str
    url:str 
    # commit_url: str 
    committed_date: str 
    additions: int 
    deletions: int 
    authored_date: str 
    message_body: str 
    message_headline: str 
    author_count: int
    committer: Optional[Committer] = None
    changed_files: Optional[int] = None
    authors: List[Committer] = None
    
@dataclass
class FileChange(Serializable): 
    additions: int 
    deletions: int 
    path: str 
    change_type: str 

@dataclass
class RepositoryPeek(Serializable): 
    id: str 
    name: str 
    url: str 
    
@dataclass
class PrTest(Serializable):
    id: str 
    title: str
    created_at: Optional[str] = None

@dataclass
class PullRequest(Serializable):
    id: str
    # database_id: int
    title: str
    url: str
    number: int 
    body: str
    state: str
    created_at: str
    is_draft: bool 
    changed_files: int 
    is_cross_repository: bool 
    locked: bool 
    is_in_merge_queue: bool 
    additions: int 
    deletions: int 
    author: Union[User, Bot, Mannequin, Organization, EnterpriseUserAccount, None, UserPeek]
    label_count: int 
    base_repository: RepositoryPeek
    head_repository: RepositoryPeek
    timeline_count: int 
    merged_at: Optional[str] = None
    closed_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_edited_at: Optional[str] = None
    published_at: Optional[str] = None
    review_decision: Optional[str] = None
    head_ref_name: Optional[str] = None
    head_ref_oid: Optional[str] = None
    timeline_items: Optional[List[str]] = None
    base_ref_name: Optional[str] = None
    base_ref_oid: Optional[str] = None
    comments_count: Optional[int] = None
    reviews_count: Optional[int] = None
    commits_count: Optional[int] = None
    files: Optional[List[FileChange]] = None
    assignees_count: Optional[int] = None
    closing_issues_count: Optional[int] = None
    author_association: Optional[str] = None
    labels: Optional[List[Label]] = None
    active_lock_reason: Optional[str] = None
    

@dataclass
class Repository(Serializable):
    id: str 
    # database_id: int
    pr_id: str
    role: str # base or head
    name: str
    name_with_owner: str 
    url: str
    ssh_url: str 
    stargazer_count: int
    is_fork: bool
    is_archived: bool 
    is_disabled: bool 
    is_empty: bool 
    is_in_organization: bool 
    is_locked: bool 
    is_private: bool 
    is_mirror: bool 
    is_template: bool 
    is_user_configuration_repository: bool 
    fork_count: int
    forking_allowed: bool 
    created_at: str 
    visibility: str 
    owner: UserPeek 
    topics_count: int 
    languages: List[str]
    language_count: int
    watchers: int
    license_info: Optional[str] = None
    default_brach: Optional[str] = None
    repository_topics: Optional[List[str]] = None
    primary_language: Optional[str] = None
    lock_reason: Optional[str] = None
    pushed_at: Optional[str] = None
    updated_at: Optional[str] = None
    archived_at: Optional[str] = None
    description: Optional[str] = None

@dataclass
class PullRequestTest(Serializable):
    id: str 
    # database_id: int 
    title: str 
    author: UserPeek
    url: str 
    body: str
    created_at: str
    is_draft: bool 
    additions: int 
    deletions: int 
    changed_files: int 
    commits: int
    comments: int
    reviews: int
    merged_at: Optional[str] = None
    closed_at: Optional[str] = None