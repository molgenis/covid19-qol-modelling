#!/usr/bin/env python3

# Imports
import pandas as pd
import numpy as np
import os
import sys
import re
sys.path.append(
    '/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/covid19-qol-modelling/src/python')
from config import get_config
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from collections import Counter
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def calculate_depressive_before(df, depressive, num_quest):
    """
    
    """
    # Sort the columns (results of depressive)
    depressive = sorted(depressive)
    # Filter dataframe on columns
    df_select = df[['project_pseudo_id'] + depressive]
    # Make lists
    list_1_2 = list()
    list_3 = list()
    list_3_a = list()
    list_3_c = list()
    list_3_other = list()
    # Loop over columns in depressive (sort)
    for i in depressive:
        num = i.split('_q_')[1]
        # Filter on question 1 and 2 
        if num.startswith('1') or num.startswith('2'):
            df_select[i] = df_select[i].astype(str).replace('2', 0).replace('2.0', 0).replace('1', 1).replace('1.0', 1).replace('nan', np.nan)
            list_1_2.append(i)
        # Filter on question 3
        elif num.startswith('3'):
            list_3.append(i)
            df_select[i] = df_select[i].astype(str).replace('2', 0).replace('2.0', 0).replace('1', 1).replace('1.0', 1).replace('nan', np.nan)
            # There are several 3a and 3c. So these have to be added together separately later.
            if '_a' in num:
                list_3_a.append(i)
            elif '_c' in num:
                if num[-1].isdigit():
                    list_3_c.append(i)
            else:
                list_3_other.append(i)
    # Add the results of questions 1 and 2 together
    df_select[f'before_{num_quest}_sum_mini_a_1_2'] = df_select.loc[:,list_1_2].sum(axis=1)
    # Add the results of questions 3a
    df_select[f'before_{num_quest}_sum_mini_a_3a'] = df_select[list_3_a].max(axis=1)
    # Add the results of questions 3c
    df_select[f'before_{num_quest}_sum_mini_a_3c'] = df_select[list_3_c].max(axis=1)
    # Extend list with 3a and 3c
    list_3_other.extend([f'before_{num_quest}_sum_mini_a_3a', f'before_{num_quest}_sum_mini_a_3c'])
    # Add the results of questions 3
    df_select[f'before_{num_quest}_sum_mini_a_3_all'] = df_select.loc[:,list_3_other].sum(axis=1)
    # Add question 1 and 2 for the all sum.
    list_3_other.extend(list_1_2)
    # Add all questions for depressive together
    df_select[f'before_{num_quest}_sum_mini_a_all'] = df_select.loc[:,list_3_other].sum(axis=1)
    # Return dataframe with only depressive answers
    return df_select


def calculate_anxiety_before(df, anxiety, num_quest):
    """
    
    """
    # Sort the columns (resulst of anxiety)
    anxiety = sorted(anxiety)
    # Filter dataframe on columns
    df_select = df[['project_pseudo_id', f'{num_quest}_date'] + anxiety]
    # Make lists
    list_1_ab = list()
    list_other = list()
    list_3 = list()
    # Loop over columns in anxiety (sort)
    for i in anxiety:
        num = i.split('_q_')[1]
        # Filter on question 1
        if num.startswith('1'):
            df_select[i] = df_select[i].astype(str).replace('2', 0).replace('2.0', 0).replace('1', 1).replace('1.0', 1).replace('nan', np.nan)
            list_1_ab.append(i)
        # Filter on question 2
        elif num.startswith('2'):
            df_select[i] = df_select[i].astype(str).replace('2', 0).replace('2.0', 0).replace('1', 1).replace('1.0', 1).replace('nan', np.nan)
            df_select[f'before_{num_quest}_sum_mini_o_2'] = df_select.loc[:,[i]].sum(axis=1)
            list_other.append(i)
        # Filter on question 3
        elif num.startswith('3'):
            list_3.append(i)
            list_other.append(i)
            df_select[i] = df_select[i].astype(str).replace('2', 0).replace('2.0', 0).replace('1', 1).replace('1.0', 1).replace('nan', np.nan)
    # Add the results of questions 1a en 1b together
    df_select[f'before_{num_quest}_sum_mini_o_1_ab'] = df_select.loc[:,list_1_ab].sum(axis=1)
    list_other.append(f'before_{num_quest}_sum_mini_o_1_ab')
    # Add the results of questions 3
    df_select[f'before_{num_quest}_sum_mini_o_3_all'] = df_select.loc[:,list_3].sum(axis=1)
    # Add the results of all the questions
    df_select[f'before_{num_quest}_sum_mini_o_all'] = df_select.loc[:,list_other].sum(axis=1)
    # Return dataframe with only anxiety answers
    return df_select


def mini_before_covid(path_myfolder):
    """
    
    """
    # Empty dataframes
    mini = pd.DataFrame(columns=['project_pseudo_id'])
    dep_all = pd.DataFrame(columns=['project_pseudo_id'])
    aux_all = pd.DataFrame(columns=['project_pseudo_id'])
    name_date_col = ''
    # Loop over different OR (1a, 2a, 3a)
    for num_quest in ['1a', '2a', '3a']:
        print('***************************')
        # Make lists and sets
        set_type_mini = set()
        # Columns with results of depressive
        depressive = list()
        # Columns with results of anxiety
        anxiety = list()
        print(num_quest)
        # Read dataframe
        df = pd.read_csv(f"{path_myfolder}df/{num_quest}_mini.tsv.gz", sep='\t', encoding='utf-8', compression='gzip')
        # Create nan values from the following values in the list 
        none_value = ['"$4"', '"$5"', '"$6"', '"$7"', '$4', '$5', '$6', '$7']
        df[df.isin(none_value)] = np.nan
        # Loop over columns of df
        for col in df.columns:
            # Check if 'mini' is in column
            if 'mini' in col:
                type_mini = col.split('_')[2]
                sort_mini = col.split('_')[1]
                set_type_mini.add(type_mini)
                if type_mini == 'a' and sort_mini =='mini':
                    depressive.append(col)
                elif type_mini == 'o'and sort_mini =='mini':
                    anxiety.append(col)
            # Check date for later filtering
            if 'date' in col:
                name_date_col = col
                df[name_date_col] = pd.to_datetime(df[name_date_col], errors='coerce')
        # Call calculate_depressive_before
        df_dep = calculate_depressive_before(df, depressive, num_quest)
        # Call calculate_anxiety_before
        df_anx = calculate_anxiety_before(df, anxiety, num_quest)
        # Only filter 3a by date as it may contain answers from during the pandemic as well
        if num_quest == '3a':
            df_date = df[['project_pseudo_id', name_date_col]]
            df_date = df_date[(df_date[name_date_col] < '2020-01')]
            df_date['3a_before_2020'] = 'yes'
        else:
            df_date = df[['project_pseudo_id', name_date_col]]
        # Merge df_date and df_anx
        aux_all = pd.merge(aux_all, df_anx, on=['project_pseudo_id'], how='outer')
        aux_all = pd.merge(df_date, aux_all, on=['project_pseudo_id'], how='outer')
        # Merge df_date and df_dep
        dep_all = pd.merge(dep_all, df_dep, on=['project_pseudo_id'], how='outer')
        dep_all = pd.merge(df_date, dep_all, on=['project_pseudo_id'], how='outer')
        # Merge df_dep, df_aux and df_date
        df_dep_anx = pd.merge(df_dep, df_anx, on=['project_pseudo_id'], how='outer')
        df_dep_anx = pd.merge(df_date, df_dep_anx, on=['project_pseudo_id'], how='outer')

    #     df_dep_anx.to_csv(f"{path_myfolder}df/{num_quest}_filter_mini_dep_aux.tsv.gz", sep='\t',
    #                     encoding='utf-8', compression='gzip', index=False)
        # Merge all the dep_anx with mini
        mini = pd.merge(mini, df_dep_anx, on=['project_pseudo_id'], how='outer')
    # Set index
    mini = mini.set_index('project_pseudo_id')
    aux_all = aux_all.set_index('project_pseudo_id')
    dep_all = dep_all.set_index('project_pseudo_id')

    # mini.to_csv(f"{path_myfolder}df/ALL_filter_mini_dep_aux.tsv.gz", sep='\t',
    #                     encoding='utf-8', compression='gzip') #, index=False
    # aux_all.to_csv(f"{path_myfolder}df/aux_filter_mini.tsv.gz", sep='\t',
    #                     encoding='utf-8', compression='gzip') #, index=False
    # dep_all.to_csv(f"{path_myfolder}df/dep_filter_mini.tsv.gz", sep='\t',
    #                     encoding='utf-8', compression='gzip') #, index=False
    # Filter on the columns with before
    # Then you have the columns with sum of questions from before the corona
    before_mini = [col for col in mini.columns if 'before' in col]
    mini[before_mini].to_csv(f"{path_myfolder}df/before_mini.tsv.gz", sep='\t',
                        encoding='utf-8', compression='gzip') #, index=False
    return mini[before_mini]


def select_label(before_mini_df, set_participants):
    """
    
    """
    # Filter dataframe on covid participants
    before_mini_df = before_mini_df[before_mini_df['project_pseudo_id'].isin(set_participants)]
    # A. MAJOR DEPRESSIVE EPISODE
    df_1_2 = before_mini_df[before_mini_df['before_2a_sum_mini_a_1_2'] >= 1]
    major_depressive_episode = df_1_2[df_1_2['before_2a_sum_mini_a_all'] >= 5]
    major_depressive_episode['major_depressive_episode'] = 1
    
    # not_major_depressive_episode = before_mini_df[~before_mini_df['project_pseudo_id'].isin(list(major_depressive_episode['project_pseudo_id']))] #df_1_2[df_1_2['before_2a_sum_mini_a_all'] < 5]
    # not_major_depressive_episode['not_major_depressive_episode'] = 1
    # not_major_depressive_episode['not_major_depressive_episode'] = not_major_depressive_episode['not_major_depressive_episode'].fillna(0)
    # print(dict(Counter(list(not_major_depressive_episode['not_major_depressive_episode']))))

    print('A. MAJOR DEPRESSIVE EPISODE')
    print(len(before_mini_df))
    print(len(df_1_2))
    print(len(major_depressive_episode))
    # print(len(not_major_depressive_episode))
    print()
    # O. GENERALIZED ANXIETY DISORDER
    df_1 = before_mini_df[before_mini_df['before_2a_sum_mini_o_1_ab'] >= 2]
    df_2 = df_1[df_1['before_2a_sum_mini_o_2'] >= 1]
    generalized_anxiety_disorder = df_2[df_2['before_2a_sum_mini_o_3_all'] >= 3]
    generalized_anxiety_disorder['generalized_anxiety_disorder'] = 1
    # not_generalized_anxiety_disorder = before_mini_df[~before_mini_df['project_pseudo_id'].isin(list(generalized_anxiety_disorder['project_pseudo_id']))] #df_2[df_2['before_2a_sum_mini_o_3_all'] < 3]
    # not_generalized_anxiety_disorder['not_generalized_anxiety_disorder'] = 1
    print('O. GENERALIZED ANXIETY DISORDER')
    print(len(before_mini_df))
    print(len(df_1))
    print(len(df_2))
    print(len(generalized_anxiety_disorder))
    # print(len(not_generalized_anxiety_disorder))
    print()

    before_mini_df = pd.merge(before_mini_df, major_depressive_episode[['project_pseudo_id', 'major_depressive_episode']], on=['project_pseudo_id'], how='outer')
    before_mini_df['major_depressive_episode'] = before_mini_df['major_depressive_episode'].fillna(0)
    print(dict(Counter(list(before_mini_df['major_depressive_episode']))))
    # before_mini_df = pd.merge(before_mini_df, not_major_depressive_episode[['project_pseudo_id', 'not_major_depressive_episode']], on=['project_pseudo_id'], how='outer')
    before_mini_df = pd.merge(before_mini_df, generalized_anxiety_disorder[['project_pseudo_id', 'generalized_anxiety_disorder']], on=['project_pseudo_id'], how='outer')
    before_mini_df['generalized_anxiety_disorder'] = before_mini_df['generalized_anxiety_disorder'].fillna(0)
    print(dict(Counter(list(before_mini_df['generalized_anxiety_disorder']))))
    # before_mini_df = pd.merge(before_mini_df, not_generalized_anxiety_disorder[['project_pseudo_id', 'not_generalized_anxiety_disorder']], on=['project_pseudo_id'], how='outer')
    return before_mini_df #major_depressive_episode, not_major_depressive_episode, generalized_anxiety_disorder, not_generalized_anxiety_disorder

def sum_same_quest(mini_df, list_cat):
    """
    
    """
    list_cat = sorted(list_cat)
    for num in list_cat:
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
    """
    
    """
    print('DEP')
    mini_above = sum_same_quest(mini_df, depressive)
    # print(mini_above)
    list_df_3 = [col for col in mini_above.columns if f'mini_a3' in col]
    # print(list_df_3)
    list_1_2 = list(set(list(mini_above.columns)) - set(list_df_3))
    # print(list_1_2)
    # Add the results of questions 1 and 2 together
    mini_above[f'between_sum_mini_a_1_2'] = mini_above.loc[:,list_1_2].sum(axis=1)
    # Add the results of questions 3
    mini_above[f'between_sum_mini_a_3_all'] = mini_above.loc[:,list_df_3].sum(axis=1)
    # Add all questions for depressive together
    mini_above[f'between_sum_mini_a_all'] = mini_above.loc[:,list_df_3 + list_1_2].sum(axis=1)
    sum_col = ['project_pseudo_id'] + [col for col in mini_above.columns if f'between_sum_mini_a' in col]
    # print(mini_above[sum_col])
    return mini_above[sum_col]

def calculate_anxiety_between(mini_df, anxiety):
    """
    
    """
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


def mini_covid(path_results, path_myfolder):
    """
    
    """
    # Empty sets and dataframes
    # set_participants = set()
    # set_cols = set()
    # mini_df = pd.DataFrame(columns=['project_pseudo_id'])
    # # Loop over questionnaire results
    # for files in os.listdir(path_results):
    #     # If file starts with 'cov'
    #     if files.startswith('cov'):
    #         filenum = files.split('_')[2]
    #         print(filenum)
    #         # Read dataframe
    #         df = pd.read_csv(f'{path_results}{files}', sep=',', encoding='utf-8')
    #         # Create nan values from the following values in the list 
    #         none_value = ['"$4"', '"$5"', '"$6"', '"$7"', '$4', '$5', '$6', '$7']
    #         df[df.isin(none_value)] = np.nan
    #         # Update set of participants
    #         set_participants.update(list(df['project_pseudo_id']))
    #         # Get columns
    #         mini_col = [col for col in df.columns if 'mini' in col]
    #         fatique = [col for col in df.columns if re.match(r'.*_fatigue_adu_q_[12]_[bad]', col)]            
    #         # Merge the selected dataframe to mini_df
    #         mini_df = pd.merge(mini_df, df[['project_pseudo_id'] + mini_col + fatique], on=['project_pseudo_id'], how='outer')
    #         # Make set of columns names
    #         for i in list(mini_col + fatique):
    #             # Replace covt{num} with ''
    #             set_cols.add(i.replace(f'cov{filenum}', ''))
    # mini_df.to_csv(f"{path_myfolder}df/between_mini.tsv.gz", sep='\t',
    #                     encoding='utf-8', compression='gzip', index=False)
    mini_df = pd.read_csv(f"{path_myfolder}df/between_mini.tsv.gz", sep='\t', encoding='utf-8', compression='gzip')
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

    # print(depressive_set)
    # print('////////')
    # print(anxiety_set)

    # mini_df = mini_df.set_index('project_pseudo_id')
    df_dep = calculate_depressive_between(mini_df, list(depressive_set))
    df_anx = calculate_anxiety_between(mini_df, anxiety_set)
    # calculate_anxiety_before(mini_df, anxiety, num_quest)
    return mini_df, set_cols, list(set_participants), df_dep, df_anx


def main():
    config = get_config()
    # Different paths
    my_folder = config['my_folder']
    path_myfolder = config['path_read_QOL']
    path_variables = config['path_questionnaire_variables']
    path_results = config['path_questionnaire_results']
    path_enumerations = config['path_questionnaire_enumerations']
    # Call mini_before_covid
    # before_mini_df = mini_before_covid(path_myfolder)
    before_mini_df = pd.read_csv(f"{path_myfolder}df/before_mini.tsv.gz", sep='\t', encoding='utf-8', compression='gzip')
    before_mini_df = before_mini_df.set_index('project_pseudo_id')
    before_mini_df = before_mini_df.reset_index(level=0)
    before_2a = ['project_pseudo_id'] + [col for col in before_mini_df.columns if f'before_2a_sum' in col]
    before_mini_df = before_mini_df[before_2a]
    print(before_mini_df)
    # Call mini_covid
    mini_df, set_cols, set_participants, df_dep, df_anx = mini_covid(path_results, path_myfolder)
    # Call select_label
    before_mini_df = select_label(before_mini_df, set_participants)
    print(before_mini_df)
    print(list(before_mini_df.columns))
    print('-----------')
    print(df_anx.columns)
    print(df_dep.columns)

    merge_before_between = pd.merge(before_mini_df, df_dep, on=['project_pseudo_id'], how='outer')
    merge_before_between = pd.merge(merge_before_between, df_anx, on=['project_pseudo_id'], how='outer')
    print('---------------------------------')
    print(merge_before_between)
    print(merge_before_between.columns)
    merge_before_between['major_depressive_episode'] = merge_before_between['major_depressive_episode'].fillna(0)
    print(dict(Counter(list(merge_before_between['major_depressive_episode']))))
    merge_before_between['generalized_anxiety_disorder'] = merge_before_between['generalized_anxiety_disorder'].fillna(0)
    print(dict(Counter(list(merge_before_between['generalized_anxiety_disorder']))))
    print(list(merge_before_between.columns))
    
    merge_before_between.to_csv(f"{path_myfolder}df/between_before_mini.tsv.gz", sep='\t',
                        encoding='utf-8', compression='gzip', index=False)
    print()

    print('DONE')


if __name__ == '__main__':
    main()

