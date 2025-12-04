"""
Crime Classification Module - Zero-Shot Learning for Police CAD Data
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")


# ==================== CONSTANTS ====================

CRIME_CATEGORIES = [
    "Theft from Motor Vehicle",
    "Stolen Vehicle",
    "Theft / Petty Theft / Shoplifting",
    "Residential Burglary",
    "Battery / Domestic Violence",
    "Simple Assault / Fight",
    "Traffic Violation / Traffic Stop / Citation",
    "DUI / Drunk Driving",
    "Shots Fired / Person with Gun",
    "Suspicious Person / Prowler",
    "911 Hangup / Open Line",
    "Disturbance / Loud Party / Neighbor Dispute",
    "Welfare Check / Mental Health Crisis",
    "Trespass / Unwanted Person",
    "Vandalism / Property Damage",
    "Narcotics / Drug Possession",
    "Prostitution / Vice",
    "Robbery / Street Robbery",
    "Alarm Call / False Alarm",
    "Officer-Initiated / On-View Activity",
    "Non-Emergency / Information Call",
    "Fire / Medical / Ambulance Request",
    "Missing Person / Runaway",
    "Juvenile / Truancy / Curfew",
    "Fraud / Identity Theft / Scam",
    "Threats / Harassment / Stalking",
    "Other / Unknown / Administrative"
]


# ==================== SETUP FUNCTIONS ====================

def setup_plotting():
    """Configure matplotlib and plotly settings"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    px.defaults.width = 1000
    px.defaults.height = 600
    pio.renderers.default = "notebook"


def load_embedder(model_name="paraphrase-MiniLM-L3-v2", max_seq_length=64):
    """
    Load sentence transformer model
    
    Args:
        model_name: Name of the sentence-transformers model
        max_seq_length: Maximum sequence length for embeddings
        
    Returns:
        SentenceTransformer model
    """
    embedder = SentenceTransformer(model_name)
    embedder.max_seq_length = max_seq_length
    return embedder


# ==================== CLASSIFICATION FUNCTIONS ====================

def classify_crime_descriptions(df, embedder, categories=CRIME_CATEGORIES, 
                                table_type_col='table_type', 
                                description_col='crime_description',
                                output_col='crime_category'):
    """
    Perform zero-shot classification on crime descriptions
    
    Args:
        df: DataFrame with crime data
        embedder: SentenceTransformer model
        categories: List of crime category labels
        table_type_col: Column name for table type
        description_col: Column name for crime descriptions
        output_col: Column name for output predictions
        
    Returns:
        DataFrame with predictions added
    """
    df = df.copy()
    df[output_col] = "pending"
    
    for table_type in df[table_type_col].unique():
        print(f"\n{'='*20} {table_type} {'='*20}")
        mask = df[table_type_col] == table_type
        subset = df.loc[mask]
        print(f"   → {len(subset):,} rows")
        
        # 1. Create embeddings
        print("Computing embeddings...")
        unique_desc, inverse = np.unique(
            subset[description_col].astype(str).values, 
            return_inverse=True
        )
        embeddings = embedder.encode(
            unique_desc.tolist(),
            batch_size=8192,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
            device='cpu',
        )
        
        # 2. Create category prototypes
        print("   → Creating category prototypes (few-shot style)...")
        proto_emb = embedder.encode(categories, normalize_embeddings=True)
        
        # 3. Zero-shot classification
        print("   → Zero-shot classification...")
        sim = cosine_similarity(embeddings, proto_emb)
        predictions_idx = sim.argmax(axis=1)
        predictions = [categories[i] for i in predictions_idx]
        
        # Map back to full data
        print("   → Adding predictions to full data...")
        full_predictions = np.array(predictions)[inverse]
        df.loc[mask, output_col] = full_predictions
    
    return df


# ==================== ANALYSIS FUNCTIONS ====================

def analyze_category(df, category, category_col='crime_category', 
                     description_col='crime_description', top_n=15):
    """
    Analyze a single crime category
    
    Args:
        df: DataFrame with classified crime data
        category: Crime category to analyze
        category_col: Column name for categories
        description_col: Column name for descriptions
        top_n: Number of top descriptions to show
    """
    subset = df[df[category_col] == category]
    total = len(subset)
    
    if total == 0:
        return
    
    print(f"\n→ {category}")
    print(f"   Total records: {total:,} ({total/len(df):.1%} of all data)")
    
    # Top descriptions
    top_desc = subset[description_col].value_counts().head(top_n)
    print(f"   Top {top_n} real descriptions:")
    for desc, count in top_desc.items():
        pct = count / total * 100
        print(f"     • {desc:45} → {count:9,} ({pct:5.1f}%)")


def sanity_check_all_categories(df, category_col='crime_category', 
                                description_col='crime_description'):
    """
    Run sanity check on all crime categories
    
    Args:
        df: DataFrame with classified crime data
        category_col: Column name for categories
        description_col: Column name for descriptions
    """
    print("="*80)
    print("SANITY CHECK: What does each crime category actually contain?")
    print("="*80)
    
    categories_by_size = df[category_col].value_counts().index
    
    for cat in categories_by_size:
        analyze_category(df, cat, category_col, description_col)


def create_summary_table(df, category_col='crime_category', 
                        description_col='crime_description'):
    """
    Create summary table with top descriptions per category
    
    Args:
        df: DataFrame with classified crime data
        category_col: Column name for categories
        description_col: Column name for descriptions
        
    Returns:
        Styled DataFrame with summary statistics
    """
    summary = []
    for cat in df[category_col].unique():
        subset = df[df[category_col] == cat]
        top3 = subset[description_col].value_counts().head(3)
        summary.append({
            'crime_category': cat,
            'total_records': len(subset),
            'top_description_1': top3.index[0] if len(top3) > 0 else "",
            'count_1': top3.iloc[0] if len(top3) > 0 else 0,
            'top_description_2': top3.index[1] if len(top3) > 1 else "",
            'count_2': top3.iloc[1] if len(top3) > 1 else 0,
            'top_description_3': top3.index[2] if len(top3) > 2 else "",
            'count_3': top3.iloc[2] if len(top3) > 2 else 0,
        })
    
    summary_df = pd.DataFrame(summary).sort_values('total_records', ascending=False)
    summary_df['% of total'] = (summary_df['total_records'] / len(df) * 100).round(2)
    summary_df.reset_index(drop=True, inplace=True)
    
    return summary_df.style.background_gradient(cmap='Blues').format({
        'total_records': '{:,}',
        'count_1': '{:,}', 'count_2': '{:,}', 'count_3': '{:,}'
    })


# ==================== VISUALIZATION FUNCTIONS ====================

def plot_top_categories_comparison(df, category_col='crime_category', 
                                   table_type_col='table_type', top_n=10):
    """
    Create side-by-side bar charts for incidents vs calls
    
    Args:
        df: DataFrame with classified crime data
        category_col: Column name for categories
        table_type_col: Column name for table type
        top_n: Number of top categories to show
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: INCIDENTS
    top_inc = df[df[table_type_col] == 'INCIDENTS'][category_col].value_counts().head(top_n)
    ax1.barh(range(len(top_inc)), top_inc.values[::-1], color='steelblue', height=0.6)
    ax1.set_yticks(range(len(top_inc)))
    ax1.set_yticklabels(top_inc.index[::-1])
    ax1.set_title(f"Top {top_n} Crime Categories — INCIDENTS (Official Reports)", 
                  fontsize=14, pad=20)
    ax1.set_xlabel("Number of Records", fontsize=12)
    ax1.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.7)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    
    # Right: CALLS FOR SERVICE
    top_call = df[df[table_type_col] == 'CALLS FOR SERVICE'][category_col].value_counts().head(top_n)
    ax2.barh(range(len(top_call)), top_call.values[::-1], color='crimson', height=0.6)
    ax2.set_yticks(range(len(top_call)))
    ax2.set_yticklabels(top_call.index[::-1])
    ax2.set_title(f"Top {top_n} Crime Categories — CALLS FOR SERVICE", 
                  fontsize=14, pad=20)
    ax2.set_xlabel("Number of Records", fontsize=12)
    ax2.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.7)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    
    # Styling
    for ax in (ax1, ax2):
        ax.tick_params(axis='x', rotation=45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.tick_params(axis='both', which='both', length=0)
    
    plt.tight_layout()
    plt.show()


def plot_time_series(df, date_col='incident_date', category_col='crime_category',
                     table_type_col='table_type', start_date='2010-01-01', 
                     end_date='2025-12-31'):
    """
    Create area chart showing crime trends over time
    
    Args:
        df: DataFrame with classified crime data
        date_col: Column name for incident date
        category_col: Column name for categories
        table_type_col: Column name for table type
        start_date: Start date for filtering
        end_date: End date for filtering
        
    Returns:
        Plotly figure
    """
    df_filtered = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)].copy()
    
    monthly = (df_filtered
               .assign(month=df_filtered[date_col].dt.to_period('M').astype(str))
               .groupby(['month', table_type_col, category_col])
               .size()
               .reset_index(name='count')
               .sort_values('month'))
    
    fig = px.area(monthly, 
                  x='month', y='count', color=category_col,
                  facet_col=table_type_col,
                  title="Crime Category Trends Over Time (Calls vs Incidents)",
                  height=600)
    fig.update_xaxes(tickangle=45)
    return fig


def plot_city_heatmap(df, top_n_cities=15, category_col='crime_category',
                     city_col='city'):
    """
    Create heatmap showing crime distribution across cities
    
    Args:
        df: DataFrame with classified crime data
        top_n_cities: Number of cities to include
        category_col: Column name for categories
        city_col: Column name for city
    """
    top_cities = df[city_col].value_counts().head(top_n_cities).index
    
    heat = (df[df[city_col].isin(top_cities)]
            .groupby([city_col, category_col])
            .size()
            .unstack(fill_value=0))
    
    # Normalize per city
    heat_norm = heat.div(heat.sum(axis=1), axis=0) * 100
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(heat_norm, cmap="YlOrRd", linewidths=.5, annot=True, fmt=".1f")
    plt.title(f"Crime Category Distribution (%) — Top {top_n_cities} Cities (Row-Normalized)", 
              fontsize=16, pad=20)
    plt.ylabel("City")
    plt.xlabel("Unified Crime Category")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_dark_figure(df, top_n=12, category_col='crime_category',
                    table_type_col='table_type'):
    """
    Visualize the 'dark figure' - calls that don't become incidents
    
    Args:
        df: DataFrame with classified crime data
        top_n: Number of categories to show
        category_col: Column name for categories
        table_type_col: Column name for table type
        
    Returns:
        DataFrame with ratios
    """
    call_only = df[df[table_type_col] == 'CALLS FOR SERVICE']
    inc_only = df[df[table_type_col] == 'INCIDENTS']
    
    dark_figure = (call_only[category_col].value_counts().head(top_n)
                   .to_frame(name="Calls")
                   .join(inc_only[category_col].value_counts().rename("Incidents"), how='left')
                   .fillna(0))
    
    dark_figure['Ratio Calls→Incidents'] = dark_figure['Incidents'] / dark_figure['Calls']
    dark_figure = dark_figure.sort_values("Ratio Calls→Incidents")
    
    dark_figure.plot(kind='barh', figsize=(12, 8), width=0.8)
    plt.title("The 'Dark Figure' — How Many Calls Become Official Incidents?", 
              pad=20, fontsize=15)
    plt.xlabel("Count (log scale)")
    plt.xscale('log')
    plt.tight_layout()
    plt.grid(True, axis='x', alpha=0.3, linestyle='--')
    plt.show()
    
    print("\nMost under-reported (calls >> incidents):")
    print(dark_figure.head(5))
    
    return dark_figure


def plot_treemap(df, category_col='crime_category', state_col='state',
                city_col='city'):
    """
    Create treemap visualization of crime distribution
    
    Args:
        df: DataFrame with classified crime data
        category_col: Column name for categories
        state_col: Column name for state
        city_col: Column name for city
        
    Returns:
        Plotly figure
    """
    treemap_data = (df
                    .groupby([state_col, city_col, category_col])
                    .size()
                    .reset_index(name='count'))
    
    total_crimes = treemap_data['count'].sum()
    treemap_data['percentage'] = (treemap_data['count'] / total_crimes * 100)
    
    fig = px.treemap(
        treemap_data,
        path=[state_col, city_col, category_col],
        values='count',
        color='count',
        color_continuous_scale='Reds',
        title="<b>National Crime Category Distribution</b><br><i>State → City → Crime Category</i>",
    )
    
    fig.update_traces(
        textinfo="label+value+percent parent",
        textposition="middle center",
        textfont=dict(family="Arial Black", size=11, color="black"),
        marker=dict(line=dict(width=2, color="white")),
        hovertemplate="<b>%{label}</b><br>Records: %{value:,}<br>Percent: %{percentParent}<extra></extra>"
    )
    
    fig.update_layout(
        margin=dict(t=100, l=10, r=10, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        title_x=0.5,
        title_font=dict(size=22, family="Arial", color="black"),
        font=dict(family="Arial", color="black"),
        height=800
    )
    
    return fig


# ==================== UTILITY FUNCTIONS ====================

def print_dataset_stats(df, city_col='city', state_col='state'):
    """Print basic dataset statistics"""
    print(f"Dataset: {len(df):,} rows | {df[city_col].nunique()} cities | {df[state_col].nunique()} states")