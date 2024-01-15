from unidecode import unidecode
import pandas as pd
id_df = pd.read_csv("Fantasy-Premier-League/data/2022-23/id_dict.csv")

understat_fname = []
for name in id_df["Understat_Name"].values:
    understat_fname.append(unidecode(name))

fpl_fname = []
for name in id_df["FPL_Name"].values:
    fpl_fname.append(unidecode(name))

id_df = id_df.assign(Understat_Name_Format=understat_fname, FPL_Name_Format=fpl_fname)

def get_fpl_name(name):
    for x in range(len(id_df["FPL_Name"].values)):
        if name in id_df["FPL_Name_Format"].values[x]:
            return id_df["FPL_Name"].values[x]

