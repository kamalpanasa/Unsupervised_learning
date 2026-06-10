import pandas as pd
from src.utils import config

def get_cluster_stats(df):
    """
    Computes average stats for each cluster.
    """
    cluster_means = df.groupby("cluster")[config.NUMERICAL_COLS].mean()
    cluster_counts = df.groupby("cluster").size().rename("count")
    stats_table = pd.concat([cluster_counts, cluster_means], axis=1)
    return stats_table

def get_cluster_personas(stats_df):
    """
    Classifies K-Means clusters into standard retail shopper personas:
    - Big Spenders (High Income, High Spending)
    - Careful/Miser (High Income, Low Spending)
    - Balanced Shoppers (Medium Income, Medium Spending)
    - Frugal/Sensible (Low Income, Low Spending)
    - Careless Spenders (Low Income, High Spending)
    """
    personas = {}
    for cid in stats_df.index:
        row = stats_df.loc[cid]
        income = row["Annual Income (k$)"]
        spending = row["Spending Score (1-100)"]
        
        if income > 70 and spending > 70:
            name = "Big Spenders / Target Customers"
            desc = "High annual income and high spending habits. Key target segment for luxury items."
        elif income > 70 and spending < 40:
            name = "Careful / Conservative Shoppers"
            desc = "High annual income but low spending score. Highly selective; value efficiency and quality."
        elif income < 40 and spending > 70:
            name = "Impulsive / Careless Spenders"
            desc = "Low annual income but high spending score. Highly responsive to marketing deals and trends."
        elif income < 40 and spending < 40:
            name = "Frugal / Budget-Conscious"
            desc = "Low annual income and low spending score. Budget-focused shoppers seeking maximum discount values."
        else:
            name = "Balanced / Standard Shoppers"
            desc = "Average annual income and average spending score. Stable, middle-market consumer segment."
            
        personas[cid] = {
            "name": name,
            "description": desc,
            "count": int(row["count"]),
            "avg_age": float(row["Age"])
        }
    return personas
