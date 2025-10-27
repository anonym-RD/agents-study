"""Analyze additions, deletions, and total changes by file category."""
import polars as pl
from pathlib import Path
from spoiler.analysis.load_hf_data_polars import TableNames, load_lazy_table_for_all_agents


# File extension categorization
EXTENSION_CATEGORIES = {
    # Programming languages (source code)
    'source_code': {
        '.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.c', '.cpp', '.cc', '.cxx', 
        '.h', '.hpp', '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.kts',
        '.scala', '.m', '.mm', '.r', '.R', '.jl', '.ex', '.exs', '.erl', '.hrl',
        '.clj', '.cljs', '.dart', '.lua', '.pl', '.pm', '.sh', '.bash', '.zsh',
        '.fish', '.ps1', '.psm1', '.groovy', '.gradle', '.vue', '.svelte',
        '.sol', '.vy', '.move', '.cairo', '.hack', '.hs', '.elm', '.ml', '.fs',
        '.vb', '.vbs', '.asm', '.s', '.f', '.f90', '.f95', '.pas', '.pp',
        '.ada', '.adb', '.ads', '.nim', '.cr', '.d', '.v', '.sv', '.vhd', '.vhdl',
    },
    
    # Markup & styling (web/UI)
    'markup_styling': {
        '.html', '.htm', '.css', '.scss', '.sass', '.less', '.styl',
        '.xml', '.svg', '.xhtml', '.xsl', '.xslt',
    },
    
    # Documentation
    'documentation': {
        '.md', '.markdown', '.rst', '.txt', '.adoc', '.asciidoc', '.org',
        '.tex', '.latex', '.rdoc', '.pod', '.man', '.1', '.2', '.3', '.4', '.5',
        '.6', '.7', '.8', '.9',
    },
    
    # Configuration & data
    'config_data': {
        '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.config',
        '.properties', '.env', '.editorconfig', '.eslintrc', '.prettierrc',
        '.babelrc', '.npmrc', '.yarnrc', '.gitignore', '.gitattributes',
        '.dockerignore', '.nvmrc', '.htaccess',
    },
    
    # Data files
    'data_files': {
        '.csv', '.tsv', '.dat', '.data', '.parquet', '.arrow', '.feather',
        '.hdf5', '.h5', '.npy', '.npz', '.pkl', '.pickle', '.rds', '.rda',
        '.mat', '.db', '.sqlite', '.sqlite3', '.sql',
    },
    
    # Lock files & dependencies
    'lock_dependency': {
        '.lock', '.lock.json', '.resolved', 
        # Common lock file names (will match via contains)
    },
    
    # Binary/compiled/generated (often large diffs but not "real" code)
    'binary_generated': {
        '.min.js', '.min.css', '.bundle.js', '.bundle.css',
        '.map', '.wasm', '.pyc', '.pyo', '.class', '.o', '.obj', '.a', '.so',
        '.dll', '.dylib', '.exe', '.out', '.jar', '.war', '.ear',
        '.snap', '.snapshot',  # Test snapshots
    },
    
    # Media & assets
    'media_assets': {
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.webp', '.avif',
        '.svg', '.pdf', '.eps', '.ai', '.psd', '.sketch', '.fig', '.xd',
        '.mp4', '.mov', '.avi', '.webm', '.mp3', '.wav', '.ogg',
        '.ttf', '.otf', '.woff', '.woff2', '.eot',
    },
    
    # Specialized formats (CAD, game assets, etc.)
    'specialized': {
        '.step', '.stp', '.stl', '.obj', '.fbx', '.gltf', '.glb',  # 3D/CAD
        '.dmm', '.dmi',  # BYOND game engine
        '.unity', '.prefab', '.asset', '.mat', '.controller', '.anim',  # Unity
        '.uasset', '.umap',  # Unreal
        '.geojson', '.kml', '.kmz', '.shp',  # GIS
        '.ipynb',  # Jupyter notebooks (mix of code and output)
        '.proto', '.thrift',  # IDL
        '.graphql', '.gql',  # GraphQL
        '.wsdl', '.xsd',  # SOAP/XML schemas
    },
    
    # Notebooks (special case - mix of code and data)
    'notebooks': {
        '.ipynb', '.Rmd', '.qmd', '.jl.md',
    },
    
    # Logs & temporary files
    'logs_temp': {
        '.log', '.tmp', '.temp', '.bak', '.backup', '.swp', '.swo', '.orig',
        '.rej', '.diff', '.patch',
    },
}


def categorize_extension(ext: str) -> str:
    """Categorize a file extension into a broader category."""
    if not ext or ext == "(no extension)":
        return "no_extension"
    
    # Normalize extension (lowercase)
    ext_lower = ext.lower()
    
    # Check each category
    for category, extensions in EXTENSION_CATEGORIES.items():
        if ext_lower in extensions:
            return category
    
    # Special handling for composite extensions (e.g., .min.js, .test.js)
    for category, extensions in EXTENSION_CATEGORIES.items():
        for known_ext in extensions:
            if ext_lower.endswith(known_ext):
                return category
    
    # If not found, categorize as unknown
    return "unknown"


def main():
    """Analyze changes by file category."""
    print("="*80)
    print("ANALYZING CHANGES BY FILE CATEGORY")
    print("="*80)
    
    print("\nLoading PR data from HuggingFace dataset...")
    df_lazy = load_lazy_table_for_all_agents(TableNames.PULL_REQUESTS)
    
    print("Extracting file-level data...")
    # Explode the files list so each file becomes a row
    files_df = df_lazy.select([
        pl.col("id").alias("pr_id"),
        pl.col("agent"),
        pl.col("files"),
    ]).explode("files").select([
        pl.col("pr_id"),
        pl.col("agent"),
        pl.col("files").struct.field("path").alias("path"),
        pl.col("files").struct.field("additions").alias("additions"),
        pl.col("files").struct.field("deletions").alias("deletions"),
        pl.col("files").struct.field("change_type").alias("change_type"),
    ])
    
    # Filter out null files (PRs without file data)
    files_df = files_df.filter(pl.col("path").is_not_null())
    
    # Extract extension
    print("Extracting file extensions...")
    files_df = files_df.with_columns([
        pl.when(pl.col("path").str.contains(r"\."))
          .then(
              pl.concat_str([
                  pl.lit("."),
                  pl.col("path").str.split("/").list.last().str.split(".").list.last()
              ])
          )
          .otherwise(pl.lit("(no extension)"))
          .alias("extension"),
    ])
    
    # Collect to apply Python categorization function
    print("Collecting data for categorization...")
    files_collected = files_df.collect()
    
    # Apply categorization
    print("Categorizing file extensions...")
    files_categorized = files_collected.with_columns([
        pl.col("extension").map_elements(categorize_extension, return_dtype=pl.Utf8).alias("category"),
        (pl.col("additions").fill_null(0) + pl.col("deletions").fill_null(0)).alias("total_changes"),
    ])
    
    # Aggregate by category
    print("Aggregating by category...")
    by_category = files_categorized.group_by("category").agg([
        pl.len().alias("num_files"),
        pl.col("additions").fill_null(0).sum().alias("total_additions"),
        pl.col("deletions").fill_null(0).sum().alias("total_deletions"),
        pl.col("total_changes").sum().alias("total_changes"),
    ]).sort("total_changes", descending=True)
    
    # Calculate overall totals for percentage
    overall_totals = by_category.select([
        pl.col("num_files").sum().alias("total_files"),
        pl.col("total_additions").sum().alias("total_additions"),
        pl.col("total_deletions").sum().alias("total_deletions"),
        pl.col("total_changes").sum().alias("total_changes"),
    ])
    
    overall_files = overall_totals["total_files"][0]
    overall_additions = overall_totals["total_additions"][0]
    overall_deletions = overall_totals["total_deletions"][0]
    overall_changes = overall_totals["total_changes"][0]
    
    # Print results
    print(f"\n{'Category':<25} {'Files':>12} {'Files %':>10} {'Additions':>15} {'Deletions':>15} {'Total Changes':>15} {'Changes %':>10}")
    print("-" * 125)
    
    for row in by_category.iter_rows(named=True):
        category = row['category']
        num_files = row['num_files']
        additions = row['total_additions']
        deletions = row['total_deletions']
        changes = row['total_changes']
        
        files_pct = 100 * num_files / overall_files if overall_files > 0 else 0
        changes_pct = 100 * changes / overall_changes if overall_changes > 0 else 0
        
        print(f"{category:<25} {num_files:>12,} {files_pct:>9.2f}% {additions:>15,} {deletions:>15,} {changes:>15,} {changes_pct:>9.2f}%")
    
    # Summary
    print("\n" + "="*125)
    print(f"{'TOTALS':<25} {overall_files:>12,} {'100.00%':>10} {overall_additions:>15,} {overall_deletions:>15,} {overall_changes:>15,} {'100.00%':>10}")
    print("="*125)
    
    # Breakdown by agent for each category
    print("\n" + "="*80)
    print("BREAKDOWN BY AGENT FOR EACH CATEGORY")
    print("="*80)
    
    categories = by_category["category"].to_list()
    
    for category in categories:
        print(f"\n{category.upper().replace('_', ' ')}:")
        print(f"{'Agent':<12} {'Files':>10} {'Additions':>15} {'Deletions':>15} {'Total Changes':>15}")
        print("-" * 70)
        
        by_agent = files_categorized.filter(pl.col("category") == category).group_by("agent").agg([
            pl.len().alias("num_files"),
            pl.col("additions").fill_null(0).sum().alias("total_additions"),
            pl.col("deletions").fill_null(0).sum().alias("total_deletions"),
            pl.col("total_changes").sum().alias("total_changes"),
        ]).sort("agent")
        
        # Sort with Human first
        human_row = by_agent.filter(pl.col("agent") == "Human")
        other_rows = by_agent.filter(pl.col("agent") != "Human").sort("agent")
        by_agent_sorted = pl.concat([human_row, other_rows]) if len(human_row) > 0 else other_rows
        
        for row in by_agent_sorted.iter_rows(named=True):
            print(f"{row['agent']:<12} {row['num_files']:>10,} {row['total_additions']:>15,} "
                  f"{row['total_deletions']:>15,} {row['total_changes']:>15,}")
    
    # High-level summary
    print("\n" + "="*80)
    print("HIGH-LEVEL SUMMARY")
    print("="*80)
    
    # Group into broader categories
    source_code_changes = by_category.filter(pl.col("category") == "source_code")["total_changes"].sum() if len(by_category.filter(pl.col("category") == "source_code")) > 0 else 0
    docs_changes = by_category.filter(pl.col("category") == "documentation")["total_changes"].sum() if len(by_category.filter(pl.col("category") == "documentation")) > 0 else 0
    config_changes = by_category.filter(pl.col("category") == "config_data")["total_changes"].sum() if len(by_category.filter(pl.col("category") == "config_data")) > 0 else 0
    markup_changes = by_category.filter(pl.col("category") == "markup_styling")["total_changes"].sum() if len(by_category.filter(pl.col("category") == "markup_styling")) > 0 else 0
    
    # Combine source + markup as "actual code"
    actual_code_changes = source_code_changes + markup_changes
    
    print(f"\n{'Actual Code (source + markup):':<40} {actual_code_changes:>15,} ({100*actual_code_changes/overall_changes:.1f}%)")
    print(f"{'  - Source Code:':<40} {source_code_changes:>15,} ({100*source_code_changes/overall_changes:.1f}%)")
    print(f"{'  - Markup/Styling:':<40} {markup_changes:>15,} ({100*markup_changes/overall_changes:.1f}%)")
    print(f"{'Documentation:':<40} {docs_changes:>15,} ({100*docs_changes/overall_changes:.1f}%)")
    print(f"{'Config/Data:':<40} {config_changes:>15,} ({100*config_changes/overall_changes:.1f}%)")
    
    non_code = overall_changes - actual_code_changes - docs_changes - config_changes
    print(f"{'Other (media, binary, specialized, etc.):':<40} {non_code:>15,} ({100*non_code/overall_changes:.1f}%)")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

