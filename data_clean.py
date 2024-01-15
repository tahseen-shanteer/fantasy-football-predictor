import os
from unidecode import unidecode
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json

season = "2023-24"
merged_gws = "Fantasy-Premier-League/data/%s/gws/merged_gw.csv"
fixtures = "Fantasy-Premier-League/data/%s/fixtures.csv"

def merged_gw_stats(season):
    global merged_gws, fixtures
    merged_gw_df = pd.read_csv(merged_gws % season)
    fixtures_df = pd.read_csv(fixtures % season)
    merged_gw_df = merged_gw_df.merge(fixtures_df[["id", "team_h_difficulty", "team_a_difficulty"]], left_on="fixture", right_on="id", how="left")
    merged_gw_df["fdr"] = np.where(merged_gw_df["was_home"], merged_gw_df["team_h_difficulty"], merged_gw_df["team_a_difficulty"])
    return merged_gw_df

def clean_merged_gw(df):
    features = ['GW', 'name', 'team', 'assists', 'bonus', 'clean_sheets', 'expected_assists', 'expected_goal_involvements', 'expected_goals', 'expected_goals_conceded', 'fdr', 'goals_conceded', 'goals_scored', 'ict_index', 'minutes', 'position', 'saves', 'total_points', 'was_home', 'xP']
    df = df[features]
    label_encoder = LabelEncoder()
    df["was_home"] = label_encoder.fit_transform(df["was_home"])
    df["position"] = label_encoder.fit_transform(df["position"])
    return df
