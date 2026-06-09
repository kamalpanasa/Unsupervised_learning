import pandas as pd
from src.utils import config

def get_cluster_stats(df):
    """
    Computes average stats and count for each cluster.
    """
    cluster_means = df.groupby("cluster")[config.STAT_COLS].mean()
    cluster_counts = df.groupby("cluster").size().rename("count")
    stats_table = pd.concat([cluster_counts, cluster_means], axis=1)
    return stats_table

def get_cluster_profiles(stats_df):
    """
    Translates numeric averages into descriptive profiles.
    """
    profiles = {}
    for cid in stats_df.index:
        row = stats_df.loc[cid]
        # Figure out dominant trait
        max_stat = row[config.STAT_COLS].idxmax()
        avg_attack = row["Attack"]
        avg_defense = row["Defense"]
        avg_speed = row["Speed"]
        
        if avg_attack > 90 and avg_speed > 90:
            name = "Sweepers / Fast Attackers"
            desc = "High Attack and Speed. Ideal for landing quick, decisive blows."
        elif avg_defense > 95:
            name = "Defensive Tanks"
            desc = "Exceptional Defense and HP. Built to withstand sustained damage."
        elif avg_speed > 90:
            name = "Speedsters"
            desc = "High Speed but lower physical durability. Relies on agility."
        elif avg_attack > 90:
            name = "Physical Brawlers"
            desc = "Heavy physical hitters, moderately paced but strong offensive stats."
        else:
            name = "Balanced Combatants"
            desc = "Well-rounded overall base stats with no severe weaknesses."
            
        profiles[cid] = {
            "name": name,
            "description": desc,
            "count": int(row["count"]),
            "top_stat": max_stat
        }
    return profiles
