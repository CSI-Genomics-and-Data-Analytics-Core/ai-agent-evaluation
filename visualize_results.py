"""
Visualization Script for GDC Evaluation Results
Analyzes and visualizes performance across different LLM frameworks and configurations

NEW METRIC DEFINITIONS:
- correctness: Binary (0 or 1) - 0 = incorrect, 1 = correct
- helpfulness: Semantic comparison (1-5 scale) - measures answer quality and relevance
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from datetime import datetime
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def parse_filename(filename):
    """Extract framework, date, and configuration from filename"""
    # Extract date
    date_match = re.search(r'\((\d{2}-[A-Za-z]{3}-\d{4})\)', filename)
    date_str = date_match.group(1) if date_match else None
    
    # Parse date
    if date_str:
        try:
            date = datetime.strptime(date_str, '%d-%b-%Y')
        except:
            date = None
    else:
        date = None
    
    # Extract framework and configuration
    if 'LLM Only' in filename:
        if 'Gemini 2.5' in filename or 'Gemini 2.5 Pro' in filename:
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
    """Load all CSV files and combine into a single dataframe"""
    csv_path = Path(csv_folder)
    all_data = []
    
    for csv_file in csv_path.glob('*.csv'):
        # Skip archive files
        if 'archive' in str(csv_file).lower():
            continue
            
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
            columns_to_keep = ['question_id', 'complexity', 'correctness', 'helpfulness',
                             'framework', 'config', 'date', 'date_str']
            available_cols = [col for col in columns_to_keep if col in df.columns]
            df_subset = df[available_cols]
            
            all_data.append(df_subset)
            print(f"Loaded: {csv_file.name} ({len(df)} rows)")
            
        except Exception as e:
            print(f"Error loading {csv_file.name}: {e}")
    
    if not all_data:
        raise ValueError("No data files loaded. Check csv folder path.")
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create combined label
    combined_df['framework_full'] = combined_df.apply(
        lambda row: f"{row['framework']} ({row['config']})", 
        axis=1
    )
    
    return combined_df


def get_custom_sort_key(framework, config):
    """Returns a sort key for consistent ordering across all plots"""
    if config == 'LLM Only':
        if framework == 'Sonnet 4.5':
            return (0, 0)
        elif framework == 'GPT 4.1':
            return (0, 1)
        elif framework == 'Gemini 2.5 Pro':
            return (0, 2)
        else:
            return (0, 99)
    elif config == 'LLM + MCP':
        return (1, 0)
    elif config == 'LLM + RAG + MCP':
        return (2, 0)
    else:
        return (99, 99)


def get_ordered_framework_labels(df):
    """Return list of framework labels ordered consistently"""
    labels = set()
    for fw, cfg in df[['framework', 'config']].drop_duplicates().values:
        labels.add(f"{fw} ({cfg})")

    desired_llm_only = [
        'Sonnet 4.5 (LLM Only)',
        'GPT 4.1 (LLM Only)',
        'Gemini 2.5 Pro (LLM Only)'
    ]

    ordered = []
    for label in desired_llm_only:
        if label in labels:
            ordered.append(label)
            labels.remove(label)

    llm_only_remaining = sorted([l for l in labels if '(LLM Only)' in l])
    ordered.extend(llm_only_remaining)
    for l in llm_only_remaining:
        labels.remove(l)

    mcp = sorted([l for l in labels if '(LLM + MCP)' in l])
    ordered.extend(mcp)
    for l in mcp:
        labels.remove(l)

    rag = sorted([l for l in labels if '(LLM + RAG + MCP)' in l])
    ordered.extend(rag)
    for l in rag:
        labels.remove(l)

    ordered.extend(sorted(labels))
    return ordered


def plot_overall_correctness(df, save_path='visualizations/overall_correctness_rate.png'):
    """Plot binary correctness rate (percentage correct) by framework"""
    score_data = []
    
    for (framework, config), group in df.groupby(['framework', 'config']):
        correctness_rate = group['correctness'].mean() * 100
        total_questions = len(group)
        correct_count = group['correctness'].sum()
        
        score_data.append({
            'framework': framework,
            'config': config,
            'correctness_rate': correctness_rate,
            'correct_count': int(correct_count),
            'total': total_questions,
            'label': f"{framework}\n({config})"
        })
    
    score_df = pd.DataFrame(score_data)
    score_df['sort_key'] = score_df.apply(
        lambda x: get_custom_sort_key(x['framework'], x['config']), 
        axis=1
    )
    score_df = score_df.sort_values('sort_key')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    config_colors = {
        'LLM Only': '#FF6B6B',
        'LLM + MCP': '#4ECDC4',
        'LLM + RAG + MCP': '#45B7D1'
    }
    colors = [config_colors.get(config, '#999999') for config in score_df['config']]
    
    bars = ax.bar(range(len(score_df)), score_df['correctness_rate'], 
                  color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Framework Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Correctness Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Correctness Rate by Framework\nPercentage of queries answered correctly', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(score_df)))
    ax.set_xticklabels(score_df['label'], rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    # Reference lines
    ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='50% threshold')
    ax.axhline(y=75, color='green', linestyle='--', alpha=0.5, linewidth=2, label='75% threshold')
    
    # Value labels
    for i, row in score_df.iterrows():
        bar = bars[i - score_df.index[0]]
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f'{height:.1f}%\n({row["correct_count"]}/{row["total"]})',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=config, alpha=0.85, edgecolor='black') 
                      for config, color in config_colors.items()]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, frameon=True)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_overall_helpfulness(df, save_path='visualizations/overall_helpfulness.png'):
    """Plot average helpfulness scores (1-5 scale)"""
    if 'helpfulness' not in df.columns:
        print("Helpfulness data not available, skipping")
        return
        
    help_data = []
    
    for (framework, config), group in df.groupby(['framework', 'config']):
        avg_helpfulness = group['helpfulness'].mean()
        
        help_data.append({
            'framework': framework,
            'config': config,
            'helpfulness': avg_helpfulness,
            'label': f"{framework}\n({config})"
        })
    
    help_df = pd.DataFrame(help_data)
    help_df['sort_key'] = help_df.apply(
        lambda x: get_custom_sort_key(x['framework'], x['config']), 
        axis=1
    )
    help_df = help_df.sort_values('sort_key')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    config_colors = {
        'LLM Only': '#FF6B6B',
        'LLM + MCP': '#4ECDC4',
        'LLM + RAG + MCP': '#45B7D1'
    }
    colors = [config_colors.get(config, '#999999') for config in help_df['config']]
    
    bars = ax.bar(range(len(help_df)), help_df['helpfulness'], 
                  color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Framework Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Helpfulness Score (1-5)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Helpfulness Score by Framework\nSemantic quality and relevance of answers (1=Poor, 5=Excellent)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(help_df)))
    ax.set_xticklabels(help_df['label'], rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 5.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Reference lines
    ax.axhline(y=2, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y=3, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y=4, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.08,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=config, alpha=0.85, edgecolor='black') 
                      for config, color in config_colors.items()]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, frameon=True)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_by_complexity(df, save_path='visualizations/by_complexity.png'):
    """Plot correctness and helpfulness by complexity level"""
    has_helpfulness = 'helpfulness' in df.columns
    n_plots = 2 if has_helpfulness else 1
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 6*n_plots))
    if n_plots == 1:
        axes = [axes]
    
    # Correctness by complexity
    correctness_data = []
    for (framework, config, complexity), group in df.groupby(['framework', 'config', 'complexity']):
        correctness_rate = group['correctness'].mean() * 100
        correctness_data.append({
            'framework': framework,
            'config': config,
            'complexity': complexity,
            'correctness_rate': correctness_rate,
            'label': f"{framework}\n({config})"
        })
    
    corr_df = pd.DataFrame(correctness_data)
    corr_df['sort_key'] = corr_df.apply(
        lambda x: get_custom_sort_key(x['framework'], x['config']), 
        axis=1
    )
    
    complexity_order = ['Low', 'Medium', 'High']
    config_colors = {
        'LLM Only': '#FF6B6B',
        'LLM + MCP': '#4ECDC4',
        'LLM + RAG + MCP': '#45B7D1'
    }
    
    # Get unique configurations for x-axis
    unique_configs = corr_df[['framework', 'config', 'sort_key']].drop_duplicates().sort_values('sort_key')
    x_labels = [f"{row['framework']}\n({row['config']})" for _, row in unique_configs.iterrows()]
    
    ax1 = axes[0]
    width = 0.25
    x = np.arange(len(unique_configs))
    
    for idx, complexity in enumerate(complexity_order):
        data = corr_df[corr_df['complexity'] == complexity].sort_values('sort_key')
        y_vals = data['correctness_rate'].values
        
        bars = ax1.bar(x + (idx - 1) * width, y_vals, width,
                      label=complexity, alpha=0.85, edgecolor='black')
        
        for bar, val in zip(bars, y_vals):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax1.set_ylabel('Correctness Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Correctness Rate by Complexity Level', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=9)
    ax1.set_ylim(0, 110)
    ax1.legend(title='Complexity', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.4)
    
    # Helpfulness by complexity
    if has_helpfulness:
        help_data = []
        for (framework, config, complexity), group in df.groupby(['framework', 'config', 'complexity']):
            avg_help = group['helpfulness'].mean()
            help_data.append({
                'framework': framework,
                'config': config,
                'complexity': complexity,
                'helpfulness': avg_help,
                'label': f"{framework}\n({config})"
            })
        
        help_df = pd.DataFrame(help_data)
        help_df['sort_key'] = help_df.apply(
            lambda x: get_custom_sort_key(x['framework'], x['config']), 
            axis=1
        )
        
        ax2 = axes[1]
        
        for idx, complexity in enumerate(complexity_order):
            data = help_df[help_df['complexity'] == complexity].sort_values('sort_key')
            y_vals = data['helpfulness'].values
            
            bars = ax2.bar(x + (idx - 1) * width, y_vals, width,
                          label=complexity, alpha=0.85, edgecolor='black')
            
            for bar, val in zip(bars, y_vals):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.08,
                        f'{val:.2f}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax2.set_ylabel('Helpfulness Score (1-5)', fontsize=12, fontweight='bold')
        ax2.set_title('Helpfulness Score by Complexity Level', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(x_labels, fontsize=9)
        ax2.set_ylim(0, 5.5)
        ax2.legend(title='Complexity', fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        ax2.axhline(y=3, color='orange', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_binary_distribution(df, save_path='visualizations/binary_correctness_distribution.png'):
    """Plot binary correctness distribution (correct vs incorrect)"""
    framework_configs = df.groupby(['framework', 'config']).size().reset_index()[['framework', 'config']]
    framework_configs['sort_key'] = framework_configs.apply(
        lambda x: get_custom_sort_key(x['framework'], x['config']), 
        axis=1
    )
    framework_configs = framework_configs.sort_values('sort_key')
    
    n_configs = len(framework_configs)
    n_rows = (n_configs + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5*n_rows))
    axes = axes.flatten() if n_configs > 1 else [axes]
    
    for idx, (_, row) in enumerate(framework_configs.iterrows()):
        framework = row['framework']
        config = row['config']
        
        data = df[(df['framework'] == framework) & (df['config'] == config)]
        
        ax = axes[idx]
        
        correct_count = (data['correctness'] == 1).sum()
        incorrect_count = (data['correctness'] == 0).sum()
        total = len(data)
        
        categories = ['Incorrect', 'Correct']
        counts = [incorrect_count, correct_count]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(categories, counts, alpha=0.85, edgecolor='black', linewidth=2, color=colors)
        
        ax.set_title(f'{framework}\n({config})', fontsize=11, fontweight='bold')
        ax.set_ylabel('Count', fontsize=10, fontweight='bold')
        ax.set_ylim(0, max(counts) * 1.2)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = (count / total) * 100 if total > 0 else 0
            ax.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                    f'{int(count)}\n({percentage:.1f}%)',
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    color='white' if height > max(counts) * 0.3 else 'black')
        
        correctness_rate = (correct_count / total * 100) if total > 0 else 0
        ax.text(0.5, 0.98, f'Correctness: {correctness_rate:.1f}%',
               transform=ax.transAxes, ha='center', va='top',
               fontsize=10, fontweight='bold', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    for idx in range(n_configs, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Correctness Distribution by Framework\nCorrect vs Incorrect Answer Counts', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_heatmaps(df, save_path='visualizations/heatmap_correctness.png'):
    """Create heatmap of correctness rates by complexity"""
    heatmap_data = []
    
    for (framework, config, complexity), group in df.groupby(['framework', 'config', 'complexity']):
        correctness_rate = group['correctness'].mean() * 100
        heatmap_data.append({
            'Framework': framework,
            'Config': config,
            'Complexity': complexity,
            'Correctness': correctness_rate
        })
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df['Framework_Label'] = heatmap_df.apply(
        lambda x: f"{x['Framework']} ({x['Config']})", axis=1
    )
    
    pivot_df = heatmap_df.pivot(index='Framework_Label', columns='Complexity', values='Correctness')
    pivot_df = pivot_df[['Low', 'Medium', 'High']]
    
    ordered_rows = get_ordered_framework_labels(df)
    pivot_df = pivot_df.reindex(ordered_rows)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='RdYlGn', 
                vmin=0, vmax=100, center=50,
                cbar_kws={'label': 'Correctness Rate (%)'},
                linewidths=2, linecolor='black', ax=ax,
                annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    
    ax.set_title('Correctness Rate by Complexity Level\nPercentage of queries answered correctly', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Complexity Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Framework Configuration', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    # Helpfulness heatmap if available
    if 'helpfulness' in df.columns:
        help_data = []
        for (framework, config, complexity), group in df.groupby(['framework', 'config', 'complexity']):
            avg_help = group['helpfulness'].mean()
            help_data.append({
                'Framework': framework,
                'Config': config,
                'Complexity': complexity,
                'Helpfulness': avg_help
            })
        
        help_df = pd.DataFrame(help_data)
        help_df['Framework_Label'] = help_df.apply(
            lambda x: f"{x['Framework']} ({x['Config']})", axis=1
        )
        
        pivot_help = help_df.pivot(index='Framework_Label', columns='Complexity', values='Helpfulness')
        pivot_help = pivot_help[['Low', 'Medium', 'High']]
        pivot_help = pivot_help.reindex(ordered_rows)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(pivot_help, annot=True, fmt='.2f', cmap='RdYlGn', 
                    vmin=1, vmax=5, center=3,
                    cbar_kws={'label': 'Helpfulness Score (1-5)'},
                    linewidths=2, linecolor='black', ax=ax,
                    annot_kws={'fontsize': 12, 'fontweight': 'bold'})
        
        ax.set_title('Helpfulness Score by Complexity Level\nSemantic quality and relevance of answers', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Complexity Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Framework Configuration', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        save_path_help = save_path.replace('correctness', 'helpfulness')
        plt.savefig(save_path_help, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path_help}")
        plt.close()


def plot_query_level_heatmaps(df, save_path_correctness='visualizations/query_heatmap_correctness.png',
                             save_path_helpfulness='visualizations/query_heatmap_helpfulness.png'):
    """
    Create detailed heatmaps with individual queries as columns (ordered Low->High complexity)
    and configurations as rows (LLM Only -> LLM+MCP -> LLM+RAG+MCP)
    """
    # Build question -> complexity mapping
    q_complex = df[['question_id', 'complexity']].drop_duplicates().set_index('question_id')['complexity'].to_dict()
    
    # Define complexity order for sorting questions
    complexity_order = {'Low': 0, 'Medium': 1, 'High': 2}
    
    # Create Framework_Label column
    df_copy = df.copy()
    df_copy['Framework_Label'] = df_copy.apply(lambda x: f"{x['framework']} ({x['config']})", axis=1)
    
    # CORRECTNESS HEATMAP
    # Pivot so rows are framework_label and columns are question_id
    pivot_corr = df_copy.pivot_table(index='Framework_Label', columns='question_id', 
                                     values='correctness', aggfunc='mean')
    
    # Order columns by complexity then question id
    cols = list(pivot_corr.columns)
    cols_sorted = sorted(cols, key=lambda q: (complexity_order.get(q_complex.get(q, 'Medium'), 1), q))
    pivot_corr = pivot_corr[cols_sorted]
    
    # Order rows using consistent ordering helper
    ordered_rows = get_ordered_framework_labels(df_copy)
    pivot_corr = pivot_corr.reindex(ordered_rows)
    
    # Plot correctness heatmap
    fig, ax = plt.subplots(figsize=(max(20, len(cols_sorted)*0.35), max(8, len(pivot_corr.index)*0.8)))
    
    # Convert to percentage for display (0-100%)
    pivot_corr_pct = pivot_corr * 100
    
    sns.heatmap(pivot_corr_pct, annot=True, fmt='.0f', cmap='RdYlGn', 
                vmin=0, vmax=100, center=50,
                cbar_kws={'label': 'Correctness Rate (%)'},
                linewidths=0.5, linecolor='gray', ax=ax,
                annot_kws={'fontsize': 8, 'fontweight': 'bold'})
    
    ax.set_title('Query-Level Correctness Rates\nPercentage correct for each question (ordered by complexity: Low → Medium → High)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Question ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Framework Configuration', fontsize=12, fontweight='bold')
    
    # Improve label readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    
    # Add vertical lines to separate complexity levels
    prev_complexity = None
    for i, col in enumerate(cols_sorted):
        curr_complexity = q_complex.get(col, 'Medium')
        if prev_complexity is not None and curr_complexity != prev_complexity:
            ax.axvline(x=i, color='blue', linewidth=2.5, linestyle='--', alpha=0.7)
        prev_complexity = curr_complexity
    
    plt.tight_layout()
    Path(save_path_correctness).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path_correctness, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path_correctness}")
    plt.close()
    
    # HELPFULNESS HEATMAP
    if 'helpfulness' in df_copy.columns:
        pivot_help = df_copy.pivot_table(index='Framework_Label', columns='question_id', 
                                         values='helpfulness', aggfunc='mean')
        
        # Order columns by complexity then question id
        pivot_help = pivot_help[cols_sorted]
        
        # Order rows using consistent ordering
        pivot_help = pivot_help.reindex(ordered_rows)
        
        # Plot helpfulness heatmap
        fig, ax = plt.subplots(figsize=(max(20, len(cols_sorted)*0.35), max(8, len(pivot_help.index)*0.8)))
        
        sns.heatmap(pivot_help, annot=True, fmt='.1f', cmap='RdYlGn', 
                    vmin=1, vmax=5, center=3,
                    cbar_kws={'label': 'Helpfulness Score (1-5)'},
                    linewidths=0.5, linecolor='gray', ax=ax,
                    annot_kws={'fontsize': 8, 'fontweight': 'bold'})
        
        ax.set_title('Query-Level Helpfulness Scores\nSemantic quality for each question (ordered by complexity: Low → Medium → High)', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Question ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('Framework Configuration', fontsize=12, fontweight='bold')
        
        # Improve label readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        
        # Add vertical lines to separate complexity levels
        prev_complexity = None
        for i, col in enumerate(cols_sorted):
            curr_complexity = q_complex.get(col, 'Medium')
            if prev_complexity is not None and curr_complexity != prev_complexity:
                ax.axvline(x=i, color='blue', linewidth=2.5, linestyle='--', alpha=0.7)
            prev_complexity = curr_complexity
        
        plt.tight_layout()
        Path(save_path_helpfulness).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_helpfulness, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path_helpfulness}")
        plt.close()


def generate_summary_report(df, save_path='visualizations/summary_report.txt'):
    """Generate comprehensive text summary report"""
    report = []
    report.append("=" * 80)
    report.append("GDC EVALUATION SUMMARY REPORT")
    report.append("=" * 80)
    report.append("")
    report.append("METRIC DEFINITIONS:")
    report.append("  - Correctness: Whether the answer is correct (0 = No, 1 = Yes)")
    report.append("  - Helpfulness: Semantic quality and relevance (1-5 scale)")
    report.append("")
    
    # Overall statistics
    report.append("OVERALL STATISTICS")
    report.append("-" * 80)
    report.append(f"Total evaluations: {len(df)}")
    report.append(f"Unique questions: {df['question_id'].nunique()}")
    report.append(f"Frameworks tested: {', '.join(df['framework'].unique())}")
    report.append(f"Configurations: {', '.join(df['config'].unique())}")
    
    overall_correctness = df['correctness'].mean() * 100
    report.append(f"Overall correctness rate: {overall_correctness:.1f}%")
    
    if 'helpfulness' in df.columns:
        overall_helpfulness = df['helpfulness'].mean()
        report.append(f"Overall helpfulness score: {overall_helpfulness:.2f}/5.00")
    report.append("")
    
    # Performance by framework
    report.append("PERFORMANCE BY FRAMEWORK AND CONFIGURATION")
    report.append("-" * 80)
    
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
        
        correctness = group['correctness'].mean() * 100
        correct_count = group['correctness'].sum()
        total = len(group)
        
        report.append(f"\n{framework} ({config}):")
        report.append(f"  Overall Correctness: {correctness:.1f}% ({int(correct_count)}/{total})")
        
        if 'helpfulness' in df.columns:
            helpfulness = group['helpfulness'].mean()
            report.append(f"  Overall Helpfulness: {helpfulness:.2f}/5.00")
        
        report.append("  By Complexity:")
        for complexity in ['Low', 'Medium', 'High']:
            comp_group = group[group['complexity'] == complexity]
            if len(comp_group) > 0:
                comp_correctness = comp_group['correctness'].mean() * 100
                comp_correct = comp_group['correctness'].sum()
                comp_total = len(comp_group)
                report.append(f"    {complexity:6s}: {comp_correctness:5.1f}% ({int(comp_correct)}/{comp_total})")
                
                if 'helpfulness' in df.columns:
                    comp_help = comp_group['helpfulness'].mean()
                    report.append(f"              Helpfulness: {comp_help:.2f}/5.00")
    
    report.append("")
    
    # Best performing
    best_framework = df.groupby(['framework', 'config'])['correctness'].mean().idxmax()
    best_correctness = df.groupby(['framework', 'config'])['correctness'].mean().max() * 100
    
    report.append("BEST PERFORMING CONFIGURATION")
    report.append("-" * 80)
    report.append(f"{best_framework[0]} ({best_framework[1]})")
    report.append(f"Correctness Rate: {best_correctness:.1f}%")
    
    if 'helpfulness' in df.columns:
        best_help_framework = df.groupby(['framework', 'config'])['helpfulness'].mean().idxmax()
        best_help = df.groupby(['framework', 'config'])['helpfulness'].mean().max()
        report.append(f"\nBest Helpfulness: {best_help_framework[0]} ({best_help_framework[1]})")
        report.append(f"Helpfulness Score: {best_help:.2f}/5.00")
    
    report.append("")
    report.append("=" * 80)
    
    # Write report
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Saved: {save_path}")
    print("\n" + '\n'.join(report))


def main():
    """Main execution function"""
    print("=" * 80)
    print("GDC EVALUATION VISUALIZATION TOOL")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading data...")
    df = load_all_data('./csv')
    print(f"\nLoaded {len(df)} total evaluations")
    print(f"Frameworks: {df['framework'].unique()}")
    print(f"Configurations: {df['config'].unique()}")
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    print()
    
    plot_overall_correctness(df)
    plot_overall_helpfulness(df)
    plot_by_complexity(df)
    plot_binary_distribution(df)
    plot_heatmaps(df)
    plot_query_level_heatmaps(df)  # New: detailed query-level heatmaps
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(df)
    
    print("\n" + "=" * 80)
    print("All visualizations generated successfully!")
    print("Check the 'visualizations' folder for results.")
    print("=" * 80)


if __name__ == "__main__":
    main()
