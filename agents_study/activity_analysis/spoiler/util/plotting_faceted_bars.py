"""Reusable plotting utilities for faceted horizontal bar charts."""
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Callable


def sort_agents_human_first(df: pl.DataFrame) -> pl.DataFrame:
    """Sort a dataframe so Human comes first, then others alphabetically."""
    human_row = df.filter(pl.col("agent") == "Human")
    other_rows = df.filter(pl.col("agent") != "Human").sort("agent")
    return pl.concat([human_row, other_rows])


def get_agents_order_reversed(df: pl.DataFrame) -> list[str]:
    """Get agent order with Human first, then reversed for horizontal bars.
    
    This ensures Human appears at the top of horizontal bar charts.
    """
    agents = ["Human"] + sorted([a for a in df["agent"].unique().to_list() if a != "Human"])
    return list(reversed(agents))


def plot_faceted_horizontal_bars(
    data: pl.DataFrame,
    categories: list[str],
    category_col: str,
    agent_col: str = "agent",
    value_col: str = "percentage",
    output_path: Optional[Path] = None,
    title_prefix: str = "",
    xlabel: str = "Percentage of PRs (%)",
    ylabel: str = "Agent",
    figsize_per_category: float = 4.0,
    fig_height: float = 6.0,
    detail_generator: Optional[Callable[[str, str], str]] = None,
    value_formatter: Callable[[float], str] = lambda v: f"{v:.1f}%",
    xlim_max: Optional[float] = None,
):
    """Create a faceted horizontal bar chart with one subplot per category.
    
    Args:
        data: DataFrame with columns [agent_col, category_col, value_col]
        categories: Ordered list of categories to plot (each gets its own facet)
        category_col: Name of the column containing categories
        agent_col: Name of the column containing agent names
        value_col: Name of the column containing values to plot
        output_path: Path to save the plot (without extension). Will save both .png and .pdf
        title_prefix: Prefix to add to the overall figure title
        xlabel: Label for x-axis
        ylabel: Label for y-axis (only shown on leftmost facet)
        figsize_per_category: Width in inches per category facet
        fig_height: Height of the figure in inches
        detail_generator: Optional function(category, agent) -> str that returns detail text
                         to display below the value label for specific category/agent combos
        value_formatter: Function to format values for display (default: "{v:.1f}%")
        xlim_max: Maximum x-axis limit. If None, auto-scales to 100 or max value * 1.1
    
    Returns:
        Figure and axes objects
    """
    print(f"\nCreating faceted horizontal bar chart...")
    
    # Get agent order (reversed so Human appears at top)
    agents_order = get_agents_order_reversed(data)
    
    # Create figure with subplots (one per category)
    n_categories = len(categories)
    fig, axes = plt.subplots(1, n_categories, figsize=(figsize_per_category * n_categories, fig_height), 
                             sharey=False)
    
    # Ensure axes is iterable even if only one subplot
    if n_categories == 1:
        axes = [axes]
    
    # Plot each category in its own subplot
    for idx, (ax, category) in enumerate(zip(axes, categories)):
        # Get data for this category
        cat_data = data.filter(pl.col(category_col) == category)
        
        # Build ordered data by agent
        agents_list = []
        values_list = []
        
        for agent in agents_order:
            agents_list.append(agent)
            # Find value for this agent in cat_data
            agent_row = cat_data.filter(pl.col(agent_col) == agent)
            if len(agent_row) > 0:
                values_list.append(agent_row[value_col].to_list()[0])
            else:
                values_list.append(0.0)
        
        agents = agents_list
        values = values_list
        
        # Create horizontal bars
        y_pos = np.arange(len(agents))
        colors = ['#FF6B6B' if agent == 'Human' else '#4ECDC4' for agent in agents]
        
        bars = ax.barh(y_pos, values, color=colors, alpha=0.8)
        
        # Customize subplot
        ax.set_xlabel(xlabel)
        ax.set_title(category, fontsize=14, fontweight='bold')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(agents)
        
        # Only show y-axis labels on leftmost plot
        if idx == 0:
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.tick_params(axis='y', labelsize=12)
        else:
            # Hide the labels but keep the ticks for alignment
            ax.set_yticklabels(['' for _ in agents])
        
        # Remove all grid lines
        ax.grid(False)
        
        # Set x-axis limits
        if xlim_max is not None:
            ax.set_xlim(0, xlim_max)
        else:
            ax.set_xlim(0, max(100, max(values) * 1.1 if values else 1))
        
        # Add value labels on bars
        for i, (bar, val, agent) in enumerate(zip(bars, values, agents)):
            if val > 0:
                label = value_formatter(val)
                
                # Add detail text if generator provided - check if it returns a tuple
                detail_lines = None
                if detail_generator is not None:
                    detail_result = detail_generator(category, agent)
                    # Handle both string and tuple returns for backward compatibility
                    if isinstance(detail_result, tuple):
                        detail_lines = detail_result
                    elif detail_result:
                        detail_lines = (detail_result, None)
                
                # Position the percentage label - move up if we have 2 lines of detail
                if detail_lines and detail_lines[1] is not None:
                    # Two lines of detail - move percentage up more
                    pct_y_offset = 0.25
                else:
                    # One line or no detail - standard position
                    pct_y_offset = 0.0
                
                ax.text(val + 1, bar.get_y() + bar.get_height()/2 + pct_y_offset,
                       label, ha='left', va='center', 
                       fontweight='bold', fontsize=11)
                
                # Add detail text lines
                if detail_lines and detail_lines[0]:
                    # First line - just below the percentage
                    ax.text(val + 1, bar.get_y() + bar.get_height()/2 + pct_y_offset - 0.2,
                           detail_lines[0], ha='left', va='center', 
                           fontsize=8, color='#666666', style='italic')
                    
                    # Second line if present
                    if detail_lines[1]:
                        ax.text(val + 1, bar.get_y() + bar.get_height()/2 + pct_y_offset - 0.39,
                               detail_lines[1], ha='left', va='center', 
                               fontsize=8, color='#666666', style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.08)
    
    # Save if path provided
    if output_path is not None:
        plt.savefig(str(output_path) + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(str(output_path) + '.pdf', bbox_inches='tight')
        print(f"  ✓ Saved to {output_path}.png and .pdf")
    
    return fig, axes


def print_distribution_table(
    data: pl.DataFrame,
    categories: list[str],
    category_col: str,
    agent_col: str = "agent",
    percentage_col: str = "percentage",
    count_col: str = "pr_count",
    title: str = "DISTRIBUTION STATISTICS",
):
    """Print a formatted table of distribution statistics.
    
    Args:
        data: DataFrame with columns [agent_col, category_col, percentage_col, count_col]
        categories: Ordered list of categories to display
        category_col: Name of the column containing categories
        agent_col: Name of the column containing agent names
        percentage_col: Name of the column containing percentage values
        count_col: Name of the column containing count values
        title: Title to display at the top
    """
    print("\n" + "="*80)
    print(title)
    print("="*80)
    
    # Get all unique agents
    all_agents = data[agent_col].unique().to_list()
    agents_order = ["Human"] + sorted([a for a in all_agents if a != "Human"])
    
    print(f"\nPercentage of PRs by Agent and {category_col.replace('_', ' ').title()}:")
    print("-" * 80)
    
    for agent in agents_order:
        agent_data = data.filter(pl.col(agent_col) == agent)
        print(f"\n{agent}:")
        for category in categories:
            cat_row = agent_data.filter(pl.col(category_col) == category)
            if len(cat_row) > 0:
                pct = cat_row[percentage_col].to_list()[0]
                count = cat_row[count_col].to_list()[0]
                print(f"  {category:25s}: {pct:5.1f}% ({count:,} PRs)")
            else:
                print(f"  {category:25s}: {0:5.1f}% (0 PRs)")

