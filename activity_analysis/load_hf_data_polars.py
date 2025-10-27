from enum import Enum
from pathlib import Path
import polars as pl
from huggingface_hub import hf_hub_download


class AgentNames(str, Enum):
    HUMAN = "Human"
    CLAUDE = "Claude"
    CODEX = "Codex"
    COPILOT = "Copilot"
    DEVIN = "Devin"
    JULES = "Jules"


class TableNames(str, Enum):
    COMMENTS = "Comments"
    COMMITS = "Commits"
    ISSUES = "Issues"
    PULL_REQUESTS = "PullRequests"
    REPOSITORIES = "Repositories"
    REVIEWS = "Reviews"


def get_table_schema(agent: AgentNames, table: TableNames):
    """Get the schema for a given agent and table."""
    return load_lazy_table(agent, table).collect_schema()


def export_all_schemas(output_file: str = "spoiler/table_schemas.txt"):
    """Export schemas for all tables to a file for easy reference.
    
    Uses Claude as the reference agent since schemas are the same across agents.
    """
    
    with open(output_file, 'w') as f:
        for table in TableNames:
            f.write(f"{'='*60}\n")
            f.write(f"Table: {table.value}\n")
            f.write(f"{'='*60}\n\n")
            
            schema = get_table_schema(AgentNames.CLAUDE, table)
            
            # Write schema in a readable format
            for col_name, col_type in schema.items():
                f.write(f"  {col_name:30s} {col_type}\n")
            
            f.write("\n\n")
    
    print(f"Schemas exported to {output_file}")


def load_lazy_table(agent: AgentNames, table: TableNames, use_local_cache: bool = True) -> pl.LazyFrame:
    """Load a table from the Hugging Face dataset.
    
    Args:
        agent: The agent name (Human, Claude, etc.)
        table: The table name (PullRequests, Reviews, etc.)
        use_local_cache: If True, downloads files to local cache using hf_hub_download
                        which provides persistent caching. If False, uses the hf:// 
                        protocol which streams data without persistent local storage.
    
    Returns a lazy DataFrame with an added 'agent' column.
    """
    if use_local_cache:
        # Download to local cache using huggingface_hub
        # This will cache the file persistently in ~/.cache/huggingface/hub/
        local_path = hf_hub_download(
            repo_id="dataset_hf",
            filename=f"data/{agent.value}/{table.value}/train-00000-of-00001.parquet",
            repo_type="dataset"
        )
        df = pl.scan_parquet(local_path)
    else:
        # Use hf:// protocol (streams data, no persistent cache for parquet files)
        path = f'hf://datasets/dataset_hf/data/{agent.value}/{table.value}/train-00000-of-00001.parquet'
        df = pl.scan_parquet(path)
    
    # Add a new column called 'agent' with the agent name
    # pl.lit() creates a literal/constant value that gets repeated for every row
    # .alias() names the column
    # with_columns() adds new columns to the dataframe
    df = df.with_columns(
        pl.lit(agent.value).alias("agent")
    )
    
    return df


def load_lazy_table_for_all_agents(table: TableNames, use_local_cache: bool = True) -> pl.LazyFrame:
    """Load and concatenate a single table across all agents in the Hugging Face dataset.
    
    Args:
        table: The table name (PullRequests, Reviews, etc.)
        use_local_cache: If True, downloads files to local cache for persistent caching
    
    Returns a lazy DataFrame with data cleaning applied:
        - For Repositories table: NULL stargazer_count values are imputed as 0
          (data collection bug: repos with 0 stars were stored as NULL)
    """
    lazy_dfs = []
    for agent in AgentNames:
        lazy_dfs.append(load_lazy_table(agent, table, use_local_cache=use_local_cache))
    
    df = pl.concat(lazy_dfs)
    
    # Apply data cleaning based on table type
    if table == TableNames.REPOSITORIES:
        # Fix: NULL stargazer_count should be 0 (data collection bug)
        # ~61% of repos have NULL star counts, but 0% have stargazer_count=0
        # This is because repos with 0 stars were stored as NULL during collection
        df = df.with_columns([
            pl.col("stargazer_count").fill_null(0)
        ])
    
    return df


def download_all_data():
    """Download all tables for all agents to local cache.
    
    This is useful to pre-populate the cache and ensure all subsequent
    operations are fast and don't require network access.
    """
    print("Downloading all data to local cache...")
    print("This may take a while on first run but will be cached for future use.\n")
    
    for table in TableNames:
        print(f"Downloading {table.value}...")
        for agent in AgentNames:
            print(f"  - {agent.value}...", end=" ", flush=True)
            # This will download and cache the file
            hf_hub_download(
                repo_id="dataset_hf",
                filename=f"data/{agent.value}/{table.value}/train-00000-of-00001.parquet",
                repo_type="dataset"
            )
            print("✓")
        print()
    
    print("All data downloaded and cached successfully!")


def main():
    # Export all table schemas
    export_all_schemas()


if __name__ == "__main__":
    main()