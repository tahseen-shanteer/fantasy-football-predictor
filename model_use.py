import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from getters import get_fpl_name
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = load_model("trained_2023-24_ players(No G-A).h5")

raw_player_data = pd.read_csv("Fantasy-Premier-League/data/2023-24/players_raw.csv")
trained_model_data = pd.read_csv("Players_2022-23.csv")

teams_df = pd.read_csv("Fantasy-Premier-League/data/2023-24/teams.csv")
fixtures_df = pd.read_csv("Fantasy-Premier-League/data/2023-24/fixtures.csv")

features = ['name', 'team', 'assists', 'bonus', 'clean_sheets', 'expected_assists', 'expected_goal_involvements', 'expected_goals', 'expected_goals_conceded', 'fdr', 'goals_conceded', 'goals_scored', 'ict_index', 'minutes', 'position', 'saves', 'total_points', 'was_home', 'xP']

pred_data = {
    'assists': 0,
    'bonus': 0,
    'clean_sheets': 0,
    'expected_assists': 0,
    'expected_goal_involvement': 0,
    'expected_goals': 0,
    'expected_goals_conceded': 0,
    'fdr': 0,
    'goals_conceded': 0,
    'goals_scored': 0,
    'ict_index': 0,
    'minutes': 0,
    'position': 0,
    'saves': 0,
    'was_home': 0,
    'xP': 0,
}

def set_player_data(name):
    global pred_data
    player_stats_df = raw_player_data[raw_player_data["first_name"] +" "+ raw_player_data["second_name"] == name]

    fdr, was_home = get_fdr(name)

    pred_data['clean_sheets'] = player_stats_df.clean_sheets_per_90.values[0]
    pred_data['bonus'] = player_stats_df.bonus.values[0] / 6
    pred_data['expected_assists'] = player_stats_df.expected_assists_per_90.values[0]
    pred_data['expected_goal_involvement'] = player_stats_df.expected_goal_involvements_per_90.values[0]
    pred_data['expected_goals'] = player_stats_df.expected_goals_per_90.values[0]
    pred_data['expected_goals_conceded'] = player_stats_df.expected_goals_conceded_per_90.values[0]
    pred_data['fdr'] = fdr
    pred_data['goals_conceded'] = player_stats_df.goals_conceded_per_90.values[0]
    pred_data['ict_index'] = player_stats_df.ict_index.values[0] / 6
    pred_data['minutes'] = player_stats_df.minutes.values[0] / 6
    pred_data['position'] = trained_model_data[trained_model_data["name"] == name].position.values[0]
    pred_data['saves'] = player_stats_df.saves_per_90.values[0]
    pred_data['was_home'] = was_home
    pred_data['xP'] = player_stats_df.ep_next.values[0]

def get_fdr(name, gw=7):
    team_id = raw_player_data[raw_player_data["first_name"] +" "+ raw_player_data["second_name"] == name].team.values[0]
    gw_df = fixtures_df[fixtures_df["event"] == gw]
    if team_id in gw_df["team_h"].values:
        return gw_df[gw_df["team_h"] == team_id].team_h_difficulty.values[0], 1
    else: 
        return gw_df[gw_df["team_a"] == team_id].team_a_difficulty.values[0], 0

relegated_team_id = [6, 12, 17]
names_list = pd.read_csv("Fantasy-Premier-League/data/2023-24/players_raw.csv")
#names_list = names_list[(names_list["total_points"] >= 30) & (~names_list['team'].isin(relegated_team_id))]
names_list["FPL_Name"] = names_list["first_name"] + " " + names_list["second_name"]
names_list = names_list['FPL_Name'].values
names_list = names_list.tolist()


player_pts = {}
processed_names = []
for name in names_list:
    try:
        set_player_data(name)
        pred_data_df = pd.DataFrame(pred_data, index=[0])
        player_pts[name] = model.predict(pred_data_df.drop(columns=["assists", "goals_scored"]))
    except IndexError:
        print("Someones name is spelt differently")
        continue

top_players = dict(sorted(player_pts.items(), key=lambda item: item[1], reverse=True))


top_players_df = pd.DataFrame({
    "Name": top_players.keys(),
    "Predicted Pts": top_players.values()
})



top_players_df.to_csv("Predictions/Top Player Predicted Pts GW7.csv")

'''
fix: the mapping of each player and they values is not mormalized because each player
has 1 row will causes std() to return 0 for all 

need to map all the players in 1 df and match the predicted pts of each to the name
'''