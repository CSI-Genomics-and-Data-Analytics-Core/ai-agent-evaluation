"""
Visualization Script for GDC Evaluation Results
Analyzes and visualizes performance across different LLM frameworks and configurations
Correctness scale: 1 (error) to 5 (correct)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def parse_filename(filename):
    """
    Extract framework, date, and configuration from filename
    
    Args:
        filename: CSV filename
        
    Returns:
        dict with framework, config, and date
    """
    # Extract date
    date_match = re.search(r'\((\d{2}-[A-Za-z]{3}-\d{4})\)', filename)
    date_str = date_match.group(1) if date_match else None
    
    # Parse date
    if date_str:
        date = datetime.strptime(date_str, '%d-%b-%Y')
    else:
        date = None
    
    # Extract framework and configuration
    if 'LLM Only' in filename:
        if 'Gemini 2.5' in filename:
            framework = 'Gemini 2.5 Pro'
            config = 'LLM Only'
        elif 'GPT 4.1' in filename:
            framework = 'GPT 4.1'
            config = 'LLM Only'
        elif 'Sonnet 4.5' in filename:
            framework = 'Sonnet 4.5'
            config = 'LLM Only'
        else:
            framework = 'Unknown'
            config = 'LLM Only'
    elif 'LLM + RAG + MCP Tools' in filename:
        framework = 'Sonnet 4.5'
        config = 'LLM + RAG + MCP'
    elif 'LLM + MCP Tools' in filename:
        framework = 'Sonnet 4.5'
        config = 'LLM + MCP'
    else:
        framework = 'Unknown'
        config = 'Unknown'
    
    return {
        'framework': framework,
        'config': config,
        'date': date,
        'date_str': date_str
    }


def load_all_data(csv_folder='./csv'):
    """
    Load all CSV files and combine into a single dataframe
    
    Args:
        csv_folder: Path to folder containing CSV files
        
    Returns:
        Combined pandas DataFrame
    """
    csv_path = Path(csv_folder)
    all_data = []
    
    for csv_file in csv_path.glob('*.csv'):
        # Parse filename
        metadata = parse_filename(csv_file.name)
        
        # Load CSV
        try:
            df = pd.read_csv(csv_file)
            
            # Add metadata columns
            df['framework'] = metadata['framework']
            df['config'] = metadata['config']
            df['date'] = metadata['date']
            df['date_str'] = metadata['date_str']
            
            # Select only needed columns
            df_subset = df[['question_id', 'complexity', 'correctness', 
                           'framework', 'config', 'date', 'date_str']]
            
            all_data.append(df_subset)
            print(f"Loaded: {csv_file.name}")
            
        except Exception as e:
            print(f"Error loading {csv_file.name}: {e}")
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create a combined label for framework + config
    combined_df['framework_full'] = combined_df.apply(
        lambda row: f"{row['framework']} ({row['config']})", 
        axis=1
    )
    
    return combined_df


def get_custom_sort_key(framework, config):
    """
    Returns a sort key for consistent ordering across all plots.
    Order: Sonnet 4.5 (LLM Only), GPT 4.1 (LLM Only), Gemini 2.5 Pro (LLM Only), 
           Sonnet 4.5 (LLM + MCP), Sonnet 4.5 (LLM + RAG + MCP)
    """
    if config == 'LLM Only':
        if framework == 'Sonnet 4.5':
            return (0, 0)
        elif framework == 'GPT 4.1':
            return (0, 1)
        elif framework == 'Gemini 2.5 Pro':
            return (0, 2)
        else:
            return (0, 99)  # Unknown LLM Only frameworks
    elif config == 'LLM + MCP':
        return (1, 0)
    elif config == 'LLM + RAG + MCP':
        return (2, 0)
    else:
        return (99, 99)  # Unknown configs


def get_ordered_framework_labels(df):
    """
    Return a list of framework label strings ("Framework (Config)") ordered consistently.
    Desired order:
      1) Sonnet 4.5 (LLM Only)
      2) GPT 4.1 (LLM Only)
      3) Gemini 2.5 Pro (LLM Only)
      4) remaining LLM Only (alphabetical)
      5) LLM + MCP
      6) LLM + RAG + MCP
    Only labels present in the dataframe are returned.
    """
    # Build set of labels present
    labels = set()
    for fw, cfg in df[['framework', 'config']].drop_duplicates().values:
        labels.add(f"{fw} ({cfg})")

    desired_llm_only = [
        'Sonnet 4.5 (LLM Only)',
        'GPT 4.1 (LLM Only)',
        'Gemini 2.5 Pro (LLM Only)'
    ]

    ordered = []
    # Add desired LLM Only in order if present
    for label in desired_llm_only:
        if label in labels:
            ordered.append(label)
            labels.remove(label)

    # Add any remaining LLM Only (alphabetical)
    llm_only_remaining = sorted([l for l in labels if '(LLM Only)' in l])
    ordered.extend(llm_only_remaining)
    for l in llm_only_remaining:
        labels.remove(l)

    # Add LLM + MCP
    mcp = sorted([l for l in labels if '(LLM + MCP)' in l])
    ordered.extend(mcp)
    for l in mcp:
        labels.remove(l)

    # Add LLM + RAG + MCP
    rag = sorted([l for l in labels if '(LLM + RAG + MCP)' in l])
    ordered.extend(rag)
    for l in rag:
        labels.remove(l)

    # Finally add any remaining labels (sorted)
    ordered.extend(sorted(labels))

    return ordered


def plot_overall_scores(df, save_path='visualizations/overall_accuracy.png'):
    """
    Plot average correctness scores by framework and configuration
    """
    # Group by framework and config
    score_data = []
    
    for (framework, config, date_str), group in df.groupby(['framework', 'config', 'date_str']):
        avg_score = group['correctness'].mean()
        score_data.append({
            'framework': framework,
            'config': config,
            'date': date_str,
            'score': avg_score,
            'label': f"{framework}\n({config})\n{date_str}"
        })
    
    score_df = pd.DataFrame(score_data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Sort using custom ordering
    score_df['sort_key'] = score_df.apply(
        lambda x: get_custom_sort_key(x['framework'], x['config']) + (x['date'],), 
        axis=1
    )
    score_df = score_df.sort_values('sort_key')
    
    # Create color map by config
    config_colors = {
        'LLM Only': '#FF6B6B',
        'LLM + MCP': '#4ECDC4',
        'LLM + RAG + MCP': '#45B7D1'
    }
    colors = [config_colors[config] for config in score_df['config']]
    
    # Create bar plot
    bars = ax.bar(range(len(score_df)), score_df['score'], color=colors, alpha=0.8, edgecolor='black')
    
    # Customize
    ax.set_xlabel('Framework Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Correctness Score (1-5 scale)', fontsize=12, fontweight='bold')
    ax.set_title('Average Correctness Score by Framework and Configuration\n(1=Error, 5=Correct)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(score_df)))
    ax.set_xticklabels(score_df['label'], rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 5.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Add horizontal reference lines
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.3, label='Error (1)')
    ax.axhline(y=3, color='orange', linestyle='--', alpha=0.3, label='Medium (3)')
    ax.axhline(y=5, color='green', linestyle='--', alpha=0.3, label='Correct (5)')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=config, alpha=0.8) 
                      for config, color in config_colors.items()]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_scores_by_complexity(df, save_path='visualizations/correctness_by_complexity.png'):
    """
    Plot average correctness scores broken down by complexity level
    """
    # Calculate scores by framework, config, and complexity
    complexity_data = []
    
    for (framework, config, date_str, complexity), group in df.groupby(
        ['framework', 'config', 'date_str', 'complexity']
    ):
        avg_score = group['correctness'].mean()
        complexity_data.append({
            'framework': framework,
            'config': config,
            'date': date_str,
            'complexity': complexity,
            'score': avg_score,
            'label': f"{framework} ({config}) {date_str}"
        })
    
    complexity_df = pd.DataFrame(complexity_data)
    
    # Add custom sort key
    complexity_df['sort_key'] = complexity_df.apply(
        lambda x: get_custom_sort_key(x['framework'], x['config']) + (x['date'],), 
        axis=1
    )
    
    # Create subplot for each complexity level
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    complexity_order = ['Low', 'Medium', 'High']
    config_colors = {
        'LLM Only': '#FF6B6B',
        'LLM + MCP': '#4ECDC4',
        'LLM + RAG + MCP': '#45B7D1'
    }
    
    for idx, complexity in enumerate(complexity_order):
        ax = axes[idx]
        data = complexity_df[complexity_df['complexity'] == complexity].copy()
        data = data.sort_values('sort_key')
        
        colors = [config_colors[config] for config in data['config']]
        bars = ax.bar(range(len(data)), data['score'], color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_title(f'{complexity} Complexity', fontsize=12, fontweight='bold')
        ax.set_ylabel('Avg Correctness Score (1-5)', fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data['label'], rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 5.5)
        ax.grid(axis='y', alpha=0.3)
        
        # Add reference lines
        ax.axhline(y=3, color='orange', linestyle='--', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Add overall legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=config, alpha=0.8) 
                      for config, color in config_colors.items()]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
              ncol=3, fontsize=10, frameon=True)
    
    fig.suptitle('Average Correctness Score by Complexity Level (1=Error, 5=Correct)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_correctness_distribution(df, save_path='visualizations/correctness_distribution.png'):
    """
    Plot distribution of correctness scores
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Get unique framework + config combinations and sort them
    framework_configs = df.groupby(['framework', 'config']).size().reset_index()[['framework', 'config']]
    framework_configs['sort_key'] = framework_configs.apply(
        lambda x: get_custom_sort_key(x['framework'], x['config']), 
        axis=1
    )
    framework_configs = framework_configs.sort_values('sort_key')
    
    for idx, (_, row) in enumerate(framework_configs.iterrows()):
        if idx >= 6:
            break
            
        framework = row['framework']
        config = row['config']
        
        data = df[(df['framework'] == framework) & (df['config'] == config)]
        
        ax = axes[idx]
        
        # Create histogram
        counts = data['correctness'].value_counts().sort_index()
        ax.bar(counts.index, counts.values, alpha=0.8, edgecolor='black', 
               color='#4ECDC4' if 'MCP' in config else '#FF6B6B')
        
        ax.set_title(f'{framework}\n({config})', fontsize=11, fontweight='bold')
        ax.set_xlabel('Correctness Score', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for x, y in zip(counts.index, counts.values):
            ax.text(x, y, str(y), ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add mean line
        mean_score = data['correctness'].mean()
        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_score:.2f}')
        ax.legend(fontsize=9)
    
    # Hide unused subplots
    for idx in range(len(framework_configs), 6):
        axes[idx].axis('off')
    
    fig.suptitle('Distribution of Correctness Scores (1=Error, 5=Correct)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_framework_comparison(df, save_path='visualizations/framework_comparison.png'):
    """
    Compare frameworks across all configurations
    """
    # Aggregate by framework and config
    comparison_data = []
    
    for (framework, config), group in df.groupby(['framework', 'config']):
        avg_score = group['correctness'].mean()
        
        # Break down by complexity
        complexity_scores = {}
        for complexity in ['Low', 'Medium', 'High']:
            comp_group = group[group['complexity'] == complexity]
            if len(comp_group) > 0:
                complexity_scores[complexity] = comp_group['correctness'].mean()
        
        comparison_data.append({
            'framework': framework,
            'config': config,
            'overall': avg_score,
            'low': complexity_scores.get('Low', 0),
            'medium': complexity_scores.get('Medium', 0),
            'high': complexity_scores.get('High', 0)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df['label'] = comparison_df.apply(
        lambda x: f"{x['framework']}\n({x['config']})", axis=1
    )
    
    # Sort using custom ordering
    comparison_df['sort_key'] = comparison_df.apply(
        lambda x: get_custom_sort_key(x['framework'], x['config']), 
        axis=1
    )
    comparison_df = comparison_df.sort_values('sort_key')
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = range(len(comparison_df))
    width = 0.2
    
    bars1 = ax.bar([i - width*1.5 for i in x], comparison_df['overall'], width, 
                   label='Overall', alpha=0.8, edgecolor='black', color='#45B7D1')
    bars2 = ax.bar([i - width*0.5 for i in x], comparison_df['low'], width, 
                   label='Low', alpha=0.8, edgecolor='black', color='#95E1D3')
    bars3 = ax.bar([i + width*0.5 for i in x], comparison_df['medium'], width, 
                   label='Medium', alpha=0.8, edgecolor='black', color='#F38181')
    bars4 = ax.bar([i + width*1.5 for i in x], comparison_df['high'], width, 
                   label='High', alpha=0.8, edgecolor='black', color='#AA96DA')
    
    ax.set_xlabel('Framework Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Correctness Score (1-5)', fontsize=12, fontweight='bold')
    ax.set_title('Framework Comparison: Overall and By Complexity\n(1=Error, 5=Correct)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['label'], rotation=45, ha='right')
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Add reference line
    ax.axhline(y=3, color='orange', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_temporal_comparison(df, save_path='visualizations/temporal_comparison.png'):
    """
    Compare performance across dates
    """
    # Build temporal dataframe with datetime objects for proper sorting
    temporal_data = []

    for _, row in df.iterrows():
        date_str = row['date_str']
        date_obj = row['date']  # parse_filename sets this as datetime or None
        if pd.isna(date_str) or date_obj is None:
            continue
        temporal_data.append({
            'date_str': date_str,
            'date_obj': date_obj,
            'framework': row['framework'],
            'config': row['config'],
            'complexity': row['complexity'],
            'score': row['correctness']
        })

    temporal_df = pd.DataFrame(temporal_data)

    if temporal_df.empty:
        print("No temporal data available")
        return

    # Aggregate by date_obj, framework, config, complexity
    temporal_agg = temporal_df.groupby(['date_obj', 'date_str', 'framework', 'config', 'complexity'])['score']\
        .mean().reset_index()

    # Ensure dates are sorted ascending (Oct then Nov)
    temporal_agg = temporal_agg.sort_values('date_obj')

    # Create subplot for each framework+config combination
    framework_configs = temporal_agg.groupby(['framework', 'config']).size().reset_index()[['framework', 'config']]
    
    # Sort using custom ordering
    framework_configs['sort_key'] = framework_configs.apply(
        lambda x: get_custom_sort_key(x['framework'], x['config']), 
        axis=1
    )
    framework_configs = framework_configs.sort_values('sort_key')
    
    n_configs = len(framework_configs)
    fig, axes = plt.subplots(n_configs, 1, figsize=(14, max(4, 3*n_configs)))

    if n_configs == 1:
        axes = [axes]

    colors = {'Low': '#95E1D3', 'Medium': '#F38181', 'High': '#AA96DA'}

    for idx, (_, row) in enumerate(framework_configs.iterrows()):
        framework = row['framework']
        config = row['config']

        data = temporal_agg[(temporal_agg['framework'] == framework) & (temporal_agg['config'] == config)]

        ax = axes[idx]

        # Plot lines for each complexity
        for complexity in ['Low', 'Medium', 'High']:
            comp_data = data[data['complexity'] == complexity].sort_values('date_obj')
            if len(comp_data) > 0:
                ax.plot(comp_data['date_obj'], comp_data['score'], 
                       marker='o', linewidth=2, markersize=8,
                       label=complexity, color=colors[complexity])

        ax.set_title(f'{framework} ({config})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10, fontweight='bold')
        ax.set_ylabel('Avg Correctness Score (1-5)', fontsize=10, fontweight='bold')
        ax.set_ylim(0, 5.5)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=3, color='orange', linestyle='--', alpha=0.3)

        # Improve x-axis formatting to avoid label overlap
        ax.xaxis.set_tick_params(rotation=30)
        # Determine unique sorted dates for this configuration
        unique_dates = sorted(data['date_obj'].unique())
        ax.set_xticks(unique_dates)
        ax.set_xticklabels([d.strftime('%d-%b-%Y') for d in unique_dates], rotation=30)

    fig.suptitle('Temporal Comparison of Correctness Scores\n(1=Error, 5=Correct)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_heatmap(df, save_path='visualizations/accuracy_heatmap.png'):
    """
    Create heatmap of average scores - separate plots for each date
    """
    # Get unique dates
    dates = sorted(df['date_str'].unique())
    
    for date_str in dates:
        if date_str is None:
            continue
            
        # Filter data for this date
        date_df = df[df['date_str'] == date_str]
        
        # Aggregate by framework+config and complexity
        heatmap_data = []
        
        for (framework, config, complexity), group in date_df.groupby(['framework', 'config', 'complexity']):
            avg_score = group['correctness'].mean()
            heatmap_data.append({
                'Framework': framework,
                'Config': config,
                'Complexity': complexity,
                'Score': avg_score
            })
        
        if len(heatmap_data) == 0:
            continue
            
        heatmap_df = pd.DataFrame(heatmap_data)
        
        # Create combined label for framework + config
        heatmap_df['Framework_Label'] = heatmap_df.apply(
            lambda x: f"{x['Framework']} ({x['Config']})", axis=1
        )
        
        # Pivot for heatmap
        pivot_df = heatmap_df.pivot(index='Framework_Label', columns='Complexity', values='Score')
        
        # Reorder columns
        pivot_df = pivot_df[['Low', 'Medium', 'High']]
        
        # Reorder rows using consistent ordering helper
        ordered_rows = get_ordered_framework_labels(date_df)
        pivot_df = pivot_df.reindex(ordered_rows)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlGn', 
                    vmin=1, vmax=5, center=3,
                    cbar_kws={'label': 'Average Correctness Score (1-5)'},
                    linewidths=1, linecolor='black', ax=ax)
        
        ax.set_title(f'Average Correctness Score Heatmap - {date_str}\n(1=Error, 5=Correct)', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Complexity Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Framework Configuration', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save with date in filename
        date_clean = date_str.replace('-', '_')
        save_path_dated = save_path.replace('.png', f'_{date_clean}.png')
        plt.savefig(save_path_dated, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path_dated}")
        plt.close()


def plot_query_level_heatmaps(df, save_path_template='visualizations/query_heatmap_{date}.png'):
    """
    Create heatmaps where columns are individual queries ordered by complexity (Low -> High).
    Generates one heatmap per date.
    """
    # Get unique dates
    dates = sorted(df['date_str'].unique())

    # Build question -> complexity mapping
    q_complex = df[['question_id', 'complexity']].drop_duplicates().set_index('question_id')['complexity'].to_dict()

    # Define complexity order for sorting questions
    complexity_order = {'Low': 0, 'Medium': 1, 'High': 2}

    for date_str in dates:
        if date_str is None:
            continue

        date_df = df[df['date_str'] == date_str]
        if date_df.empty:
            continue

        # Pivot so rows are framework_label and columns are question_id
        date_df['Framework_Label'] = date_df.apply(lambda x: f"{x['framework']} ({x['config']})", axis=1)
        pivot = date_df.pivot_table(index='Framework_Label', columns='question_id', values='correctness', aggfunc='mean')

        # Order columns by complexity then question id
        cols = list(pivot.columns)
        cols_sorted = sorted(cols, key=lambda q: (complexity_order.get(q_complex.get(q, 'Medium'), 1), q))
        pivot = pivot[cols_sorted]

        # Order rows using consistent ordering helper
        ordered_rows = get_ordered_framework_labels(date_df)
        pivot = pivot.reindex(ordered_rows)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(max(12, len(cols_sorted)*0.25 + 6), max(6, len(pivot.index)*0.5 + 4)))
        sns.heatmap(pivot, annot=False, cmap='RdYlGn', vmin=1, vmax=5, center=3,
                    cbar_kws={'label': 'Correctness (1-5)'}, linewidths=0.3, linecolor='gray', ax=ax)

        ax.set_title(f'Query-level Correctness Heatmap - {date_str}\n(columns ordered by complexity Low->High)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Question ID (Low -> High)')
        ax.set_ylabel('Framework Configuration')

        # Improve label readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

        plt.tight_layout()
        save_path = save_path_template.format(date=date_str.replace('-', '_'))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()


def plot_query_heatmap(df, save_path='visualizations/query_heatmap.png'):
    """
    Create heatmap with individual queries as columns - separate plots for each date
    """
    # Get unique dates
    dates = sorted(df['date_str'].unique())
    
    for date_str in dates:
        if date_str is None:
            continue
            
        # Filter data for this date
        date_df = df[df['date_str'] == date_str]
        
        # Create pivot table with framework+config as rows and question_id as columns
        pivot_data = []
        
        for (framework, config, question_id), group in date_df.groupby(['framework', 'config', 'question_id']):
            avg_score = group['correctness'].mean()
            complexity = group['complexity'].iloc[0]
            pivot_data.append({
                'Framework': framework,
                'Config': config,
                'Question': question_id,
                'Complexity': complexity,
                'Score': avg_score
            })
        
        if len(pivot_data) == 0:
            continue
            
        pivot_df = pd.DataFrame(pivot_data)
        
        # Create combined label for framework + config
        pivot_df['Framework_Label'] = pivot_df.apply(
            lambda x: f"{x['Framework']} ({x['Config']})", axis=1
        )
        
        # Create pivot table
        heatmap_df = pivot_df.pivot(index='Framework_Label', columns='Question', values='Score')
        
        # Sort columns by complexity (Low -> Medium -> High) and then by question ID
        # Get complexity for each question
        question_complexity = pivot_df[['Question', 'Complexity']].drop_duplicates().set_index('Question')['Complexity'].to_dict()
        
        # Define complexity order
        complexity_order = {'Low': 1, 'Medium': 2, 'High': 3}
        
        # Sort columns by complexity first, then by question ID
        sorted_columns = sorted(heatmap_df.columns, 
                               key=lambda x: (complexity_order.get(question_complexity.get(x, 'Medium'), 2), x))
        heatmap_df = heatmap_df[sorted_columns]
        
        # Order rows using consistent ordering helper
        ordered_rows = get_ordered_framework_labels(date_df)
        heatmap_df = heatmap_df.reindex(ordered_rows)
        
        # Create heatmap with larger figure size to accommodate all queries
        fig, ax = plt.subplots(figsize=(max(20, len(sorted_columns) * 0.5), 10))
        
        sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='RdYlGn', 
                    vmin=1, vmax=5, center=3,
                    cbar_kws={'label': 'Correctness Score (1-5)'},
                    linewidths=0.5, linecolor='gray', ax=ax,
                    annot_kws={'fontsize': 8})
        
        ax.set_title(f'Correctness Score by Query - {date_str}\n(1=Error, 5=Correct)', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Question ID (sorted by complexity: Low → Medium → High)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Framework Configuration', fontsize=12, fontweight='bold')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
        
        # Add vertical lines to separate complexity levels
        # Find where complexity changes
        prev_complexity = None
        for i, col in enumerate(sorted_columns):
            curr_complexity = question_complexity.get(col, 'Medium')
            if prev_complexity is not None and curr_complexity != prev_complexity:
                ax.axvline(x=i, color='blue', linewidth=2, linestyle='--', alpha=0.5)
            prev_complexity = curr_complexity
        
        plt.tight_layout()
        
        # Save with date in filename
        date_clean = date_str.replace('-', '_')
        save_path_dated = save_path.replace('.png', f'_{date_clean}.png')
        plt.savefig(save_path_dated, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path_dated}")
        plt.close()


def generate_summary_report(df, save_path='visualizations/summary_report.txt'):
    """
    Generate text summary report
    """
    report = []
    report.append("=" * 80)
    report.append("GDC EVALUATION SUMMARY REPORT")
    report.append("Correctness Scale: 1 (Error) to 5 (Correct)")
    report.append("=" * 80)
    report.append("")
    
    # Overall statistics
    report.append("OVERALL STATISTICS")
    report.append("-" * 80)
    report.append(f"Total evaluations: {len(df)}")
    report.append(f"Total unique questions: {df['question_id'].nunique()}")
    report.append(f"Frameworks tested: {df['framework'].nunique()}")
    report.append(f"Configurations: {df['config'].nunique()}")
    report.append(f"Evaluation dates: {df['date_str'].nunique()}")
    report.append(f"Overall average correctness score: {df['correctness'].mean():.2f}/5.00")
    report.append("")
    
    # Performance by framework and configuration
    report.append("PERFORMANCE BY FRAMEWORK AND CONFIGURATION")
    report.append("-" * 80)
    report.append("")
    
    # Sort framework/config combinations
    framework_configs = df.groupby(['framework', 'config']).size().reset_index()[['framework', 'config']]
    framework_configs['sort_key'] = framework_configs.apply(
        lambda x: get_custom_sort_key(x['framework'], x['config']), 
        axis=1
    )
    framework_configs = framework_configs.sort_values('sort_key')
    
    for _, row in framework_configs.iterrows():
        framework = row['framework']
        config = row['config']
        group = df[(df['framework'] == framework) & (df['config'] == config)]
        overall_score = group['correctness'].mean()
        report.append(f"{framework} ({config}):")
        report.append(f"  Overall Score: {overall_score:.2f}/5.00")
        
        for complexity in ['Low', 'Medium', 'High']:
            comp_group = group[group['complexity'] == complexity]
            if len(comp_group) > 0:
                comp_score = comp_group['correctness'].mean()
                report.append(f"  {complexity} Complexity: {comp_score:.2f}/5.00")
        report.append("")
    
    # Best performing configuration
    best_config = df.groupby(['framework', 'config'])['correctness'].mean().idxmax()
    best_score = df.groupby(['framework', 'config'])['correctness'].mean().max()
    report.append("BEST PERFORMING CONFIGURATION")
    report.append("-" * 80)
    report.append(f"{best_config[0]} ({best_config[1]}): {best_score:.2f}/5.00")
    report.append("")
    
    # Score by complexity level (averaged across all frameworks)
    report.append("AVERAGE SCORE BY COMPLEXITY LEVEL (ACROSS ALL FRAMEWORKS)")
    report.append("-" * 80)
    for complexity in ['Low', 'Medium', 'High']:
        comp_group = df[df['complexity'] == complexity]
        comp_score = comp_group['correctness'].mean()
        report.append(f"{complexity}: {comp_score:.2f}/5.00")
    report.append("")
    
    # Score distribution
    report.append("OVERALL CORRECTNESS SCORE DISTRIBUTION")
    report.append("-" * 80)
    for score in range(1, 6):
        count = (df['correctness'] == score).sum()
        percentage = count / len(df) * 100
        report.append(f"Score {score}: {count} ({percentage:.1f}%)")
    report.append("")
    
    report.append("=" * 80)
    
    # Write report
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Saved: {save_path}")
    print("\n" + '\n'.join(report))


def main():
    """
    Main execution function
    """
    print("=" * 80)
    print("GDC EVALUATION VISUALIZATION TOOL")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading data...")
    df = load_all_data('./csv')
    print(f"\nLoaded {len(df)} total evaluations")
    print(f"Unique questions: {df['question_id'].nunique()}")
    print(f"Frameworks: {df['framework'].nunique()}")
    print(f"Date range: {df['date_str'].unique()}")
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    print()
    
    plot_scores_by_complexity(df)
    plot_correctness_distribution(df)
    plot_framework_comparison(df)
    plot_temporal_comparison(df)
    plot_heatmap(df)
    # Per-query heatmaps (columns = question_id ordered Low->High)
    plot_query_level_heatmaps(df)
    plot_query_heatmap(df)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(df)
    
    print("\n" + "=" * 80)
    print("All visualizations generated successfully!")
    print("Check the 'visualizations' folder for results.")
    print("=" * 80)


if __name__ == "__main__":
    main()
