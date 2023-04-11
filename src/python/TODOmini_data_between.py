#!/usr/bin/env python3

# Imports
import pandas as pd
import numpy as np
import os
import sys
import re
sys.path.append(
    '/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/COVID_Anne')
from config import get_config
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from collections import Counter
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')




def sum_same_quest(mini_df, list_cat):
    """
    
    Selects everyone who has answered yes to the mini question 50% or more times
    """
    list_cat = sorted(list_cat)
    for num in list_cat:
        print(num)
        if 'fatigue' not in num:
            mini_col = [col for col in mini_df.columns if f'mini{num}' in col]
            mini_df[mini_col] = mini_df[mini_col].astype(str).replace('2', 0).replace('2.0', 0).replace('1', 1).replace('1.0', 1).replace('nan', np.nan)
            mini_df[f'between_mini_{num}_1'] = mini_df[mini_col].sum(axis=1)
            mini_df[f'between_mini_{num}_0'] = (mini_df[mini_col] == 0).sum(axis=1)
            mini_df[f'between_mini_{num}_percent_1']  = mini_df[f'between_mini_{num}_1'] / (mini_df[f'between_mini_{num}_1'] + mini_df[f'between_mini_{num}_0']) * 100
            mini_df[f'between_above_mini_{num}'] = np.where(mini_df[f'between_mini_{num}_percent_1'] >= 50, 1, 0)
        else:
            frag_col = [col for col in mini_df.columns if num in col]
            # for i in frag_col:
            #     print(set(list(frag_col[frag_col[i].notna()][i])))
        # print(mini_df[mini_col + [f'between_mini_{num}_1', f'between_mini_{num}_0', f'between_mini_{num}_percent_1', f'between_above_mini_{num}']])
    mini_col_above = ['project_pseudo_id'] + [col for col in mini_df.columns if f'between_above_mini_' in col]
    mini_above = mini_df[mini_col_above]
    return mini_above


def calculate_depressive_between(mini_df, depressive):
    print('DEP')
    mini_above = sum_same_quest(mini_df, depressive)
    # # print(mini_above)
    # list_df_3 = [col for col in mini_above.columns if f'mini_a3' in col]
    # # print(list_df_3)
    # list_1_2 = list(set(list(mini_above.columns)) - set(list_df_3))
    # # print(list_1_2)
    # # Add the results of questions 1 and 2 together
    # mini_above[f'between_sum_mini_a_1_2'] = mini_above.loc[:,list_1_2].sum(axis=1)
    # # Add the results of questions 3
    # mini_above[f'between_sum_mini_a_3_all'] = mini_above.loc[:,list_df_3].sum(axis=1)
    # # Add all questions for depressive together
    # mini_above[f'between_sum_mini_a_all'] = mini_above.loc[:,list_df_3 + list_1_2].sum(axis=1)
    # sum_col = ['project_pseudo_id'] + [col for col in mini_above.columns if f'between_sum_mini_a' in col]
    # # print(mini_above[sum_col])
    # return mini_above[sum_col]

def calculate_anxiety_between(mini_df, anxiety):
    list_3b = [col for col in mini_df.columns if f'minia3b' in col]
    list_3f = [col for col in mini_df.columns if f'minia3f' in col]
    for value in list_3b + list_3f:
        value = value.split('_')[1].replace('mini', '') #re.sub(r"covt.*_m", "m", value)
        anxiety.add(value) #covt\d*_=
    mini_above = sum_same_quest(mini_df, anxiety)
    # print(mini_above)
    # print(mini_above.columns)
    list_1_ab = [col for col in mini_above.columns if f'_mini_o1' in col]
    # print(list_1_ab)
    list_2 = [col for col in mini_above.columns if f'mini_o2' in col]
    # print(list_2)
    # TODO fatique toevoegen
    list_3_without = [col for col in mini_above.columns if f'mini_a3b' in col]
    list_3_without = list_3_without + [col for col in mini_above.columns if f'mini_a3f' in col]
    list_3_without = list_3_without + [col for col in mini_above.columns if f'mini_o3' in col]
    # print(list_3_without)
    # Add the results of questions 1 
    mini_above[f'between_sum_mini_o_1ab'] = mini_above.loc[:,list_1_ab].sum(axis=1)
    # Add the results of questions 2
    mini_above[f'between_sum_mini_o_2'] = mini_above.loc[:,list_2].sum(axis=1)
    # Add the results of questions 3
    mini_above[f'between_sum_mini_o_3_all'] = mini_above.loc[:,list_3_without].sum(axis=1)
    sum_col = ['project_pseudo_id'] + [col for col in mini_above.columns if f'between_sum_mini_o' in col]
    # print(mini_above[sum_col])
    return mini_above[sum_col]


def make_mini_df(path_results, mini_path):
    # Empty sets and dataframes
    set_participants = set()
    set_cols = set()
    mini_df = pd.DataFrame(columns=['project_pseudo_id'])
    # Loop over questionnaire results
    for files in os.listdir(path_results):
        # If file starts with 'cov'
        if files.startswith('cov'):
            filenum = files.split('_')[2]
            print(filenum)
            # Read dataframe
            df = pd.read_csv(f'{path_results}{files}', sep=',', encoding='utf-8')
            # Create nan values from the following values in the list 
            none_value = ['"$4"', '"$5"', '"$6"', '"$7"', '$4', '$5', '$6', '$7']
            df[df.isin(none_value)] = np.nan
            # Update set of participants
            set_participants.update(list(df['project_pseudo_id']))
            # Get columns
            mini_col = [col for col in df.columns if 'mini' in col]
            fatique = [col for col in df.columns if re.match(r'.*_fatigue_adu_q_[12]_[bad]', col)]            
            # Merge the selected dataframe to mini_df
            mini_df = pd.merge(mini_df, df[['project_pseudo_id'] + mini_col + fatique], on=['project_pseudo_id'], how='outer')
            # Make set of columns names
            for i in list(mini_col + fatique):
                # Replace covt{num} with ''
                set_cols.add(i.replace(f'cov{filenum}', ''))
    mini_df.to_csv(f"{mini_path}between_mini.tsv.gz", sep='\t',
                        encoding='utf-8', compression='gzip', index=False)
    return mini_df
    
def mini_covid(mini_path, mini_df):
    if mini_df.empty:
        mini_df = pd.read_csv(f"{mini_path}between_mini.tsv.gz", sep='\t', encoding='utf-8', compression='gzip')
    set_participants = set(mini_df['project_pseudo_id'])
    set_cols = set()
    for col in mini_df.columns:
        if re.match(r'.*_fatigue_adu_q_[12]_[bad]', col):
            # Replace covt{num} with ''
            set_cols.add(re.sub(r"covt.*_f", "f", col))
        if 'mini' in col:
            # Replace covt{num} with ''
            set_cols.add(re.sub(r"covt.*_m", "m", col)) #covt\d*_

    # Empty sets and lists
    set_type_mini = set()
    depressive = list()
    depressive_set = set()
    anxiety = list()
    anxiety_set = set()
    # Loop over columns mini_df
    for col in mini_df.columns:
        # Check if 'mini' is in column
        if 'mini' in col:
            type_mini = col.split('mini')[1].split('_')[0]
            set_type_mini.add(type_mini)
            # Check if 'a' in column Depressive
            if re.match(r'^a.*', type_mini):
                depressive.append(col)
                depressive_set.add(type_mini)
            # Check if 'o' in column Anxiety
            elif re.match(r'^o.*', type_mini):
                anxiety.append(col)
                anxiety_set.add(type_mini)
        # Check if 'fatigue' is in column
        if 'fatigue' in col:
            fatigue = f"f{col.split('_f')[1]}"
            anxiety.append(col)
            anxiety_set.add(fatigue)

    print()
    print(depressive_set)
    print('////////')
    print(anxiety_set)

    # mini_df = mini_df.set_index('project_pseudo_id')
    df_dep = calculate_depressive_between(mini_df, list(depressive_set))
    # df_anx = calculate_anxiety_between(mini_df, anxiety_set)
    # # calculate_anxiety_before(mini_df, anxiety, num_quest)
    # return mini_df, set_cols, list(set_participants), df_dep, df_anx


def main():
    config = get_config()
    # Different paths
    my_folder = config['my_folder']
    mini_path = config['MINI']
    path_myfolder = config['path_read_QOL']
    path_variables = config['path_questionnaire_variables']
    path_results = config['path_questionnaire_results']
    path_enumerations = config['path_questionnaire_enumerations']
    mini_df = pd.DataFrame()
    
    # Call mini_covid
    mini_df = make_mini_df(path_results, mini_path)
    mini_covid(mini_path, mini_df)
    # Call select_label
    
    print('DONE')


if __name__ == '__main__':
    main()

