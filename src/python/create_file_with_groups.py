#!/usr/bin/env python3

# ---------------------------------------------------------
# Author: Anne van Ewijk
# University Medical Center Groningen / Department of Genetics
#
# Copyright (c) Anne van Ewijk, 2023
#
# ---------------------------------------------------------

# Imports
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(
    '/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/covid19-qol-modelling/src/python')
from config import get_config
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import warnings

warnings.filterwarnings('ignore')


def merge_questdata_with_other(data_QOL, create_file_with_groups_path, question_15_or_more_path):
    """
    Merge different dataframes
    """
    # Read files
    df_quest = pd.read_csv(f'{question_15_or_more_path}num_quest_1_15_or_more.tsv.gz', sep='\t', encoding='utf-8',
                           compression='gzip')
    df_well = pd.read_csv(f'{question_15_or_more_path}num_quest_1_filter.tsv.gz', sep='\t', encoding='utf-8',
                          compression='gzip')
    # Read files and rename columns
    wellbeing_data = pd.read_csv(f'{data_QOL}wellbeing_data_collection.csv')
    wellbeing_data.rename(columns={'date': 'responsedate'}, inplace=True)
    household_data = pd.read_csv(f'{data_QOL}household_categories_2022_08_08.csv')
    household_data.rename(columns={'PROJECT_PSEUDO_ID': 'project_pseudo_id'}, inplace=True)
    # Merge files
    df_QOL = pd.merge(df_quest, household_data, on=['project_pseudo_id'], how='left')
    df_well = pd.merge(df_well, wellbeing_data, on=['responsedate'], how='outer')
    df_QOL = df_QOL.loc[:, ~df_QOL.columns.str.match("Unnamed")]
    df_well.to_csv(f'{create_file_with_groups_path}num_quest_1_filter_plus_wellbeing.tsv.gz', sep='\t',
                   encoding='utf-8',
                   compression='gzip', index=False)
    # Write dataframe to tsv file
    df_QOL.to_csv(f'{create_file_with_groups_path}merge_15ormore_household.tsv.gz', sep='\t', encoding='utf-8',
                  compression='gzip', index=False)


def add_media_data(path_directory, create_file_with_groups_path):
    """
    Add media questions
    """
    # Read file
    df_QOL = pd.read_csv(f'{create_file_with_groups_path}merge_15ormore_household.tsv.gz', sep='\t', encoding='utf-8',
                         compression='gzip')
    set_letter_cat = {'a': 'mediacategory_media',
                      'b': 'mediacategory_health_authorities',
                      'c': 'mediacategory_social_media',
                      'd': 'mediacategory_family_and_friends',
                      'e': 'mediacategory_other'}
    interesting_column = 'information_adu_q'
    select_columns_dict = dict()
    set_participants = set()

    for i in range(1, 10):
        df = pd.read_csv(f'{path_directory}covq_q_t0{i}_results.csv', sep=',', encoding='utf-8')
        # Replace none_values with np.nan in df
        none_value = ['"$4"', '"$5"', '"$6"', '"$7"', '$4', '$5', '$6', '$7']
        df[df.isin(none_value)] = np.nan
        for letter, value in set_letter_cat.items():
            select_columns = ['project_pseudo_id']
            # Loop over column names of all_quest
            for col in df.columns:
                # Check if 'interesting_column' in column name
                if interesting_column in col and col.endswith(letter):
                    # Add column to list
                    select_columns.append(col)
                    select_df = df[select_columns]
                    set_true = set(select_df[select_df[col].notnull()]['project_pseudo_id'])
                    # Update set
                    set_participants.update(select_df['project_pseudo_id'])
                    if letter in select_columns_dict:
                        select_columns_dict[letter]['true'].update(set_true)
                    else:
                        select_columns_dict[letter] = {'true': set_true, 'false': set()}

    dict_letter_list_participants = dict()
    # Loop over participants
    for participant in df_QOL['project_pseudo_id']:
        # Loop over dict: select_columns_dict
        for letter, value in select_columns_dict.items():
            # Check if letter in dict
            if letter not in dict_letter_list_participants:
                dict_letter_list_participants[letter] = list()
            # Check if participant in list value['true']
            if participant in value['true']:
                dict_letter_list_participants[letter].append('TRUE')
            else:
                dict_letter_list_participants[letter].append('FALSE')

    # Loop over dict: dict_letter_list_participants
    for letter, value in dict_letter_list_participants.items():
        df_QOL[set_letter_cat[letter]] = value

    # Write dataframe to tsv file
    df_QOL.to_csv(f'{create_file_with_groups_path}merge_15ormore_household_media.tsv.gz', sep='\t',
                  encoding='utf-8', compression='gzip', index=False)
    # Call sep_data: also save the different categories in a separate file
    sep_data(df_QOL, create_file_with_groups_path, 'media')


def add_general_health(create_file_with_groups_path, path_directory, config):
    """
    Add general health
    """
    # Read file
    df_QOL = pd.read_csv(f'{create_file_with_groups_path}merge_15ormore_household_media.tsv.gz', sep='\t',
                         encoding='utf-8',
                         compression='gzip')

    interesting_column = 'rand01_adu_q_'
    general_health_column = dict()
    df_gen = pd.DataFrame(columns=['project_pseudo_id'])
    # Get the general health questions
    for i in range(1, 10):
        dict_ans = dict()
        df = pd.read_csv(f'{path_directory}covq_q_t0{i}_results.csv', sep=',', encoding='utf-8')
        # Replace none_values with np.nan in df
        none_value = ['"$4"', '"$5"', '"$6"', '"$7"', '$4', '$5', '$6', '$7']
        df[df.isin(none_value)] = np.nan
        for col in df.columns:
            if interesting_column in col:
                general_health_column[str(i)] = col
                variables_df = pd.read_csv(f"{config['path_questionnaire_enumerations']}covq_q_t0{i}_enumerations.csv",
                                           sep=',', encoding='utf-8')
                select_var = variables_df[variables_df['variable_name'] == col]
                for index, row in select_var.iterrows():
                    dict_ans[str(row['enumeration_code'])] = row['enumeration_en']
                df[col] = df[col].map(dict_ans)
                # Merge files
                df_gen = pd.merge(df_gen, df[['project_pseudo_id', col]], on=['project_pseudo_id'], how="outer")
    # First question as general_health
    df_gen['general_health'] = df_gen['covt01_rand01_adu_q_1']
    # Update with newer question
    df_gen['general_health'].update(df_gen[df_gen['covt02_rand01_adu_q_1'].notnull()]['covt02_rand01_adu_q_1'])
    # Merge dataframes
    df_QOL = pd.merge(df_QOL, df_gen[['project_pseudo_id', 'general_health']], on=['project_pseudo_id'], how="left")
    # Write dataframe to tsv file
    df_QOL.to_csv(f'{create_file_with_groups_path}merge_15ormore_household_media_generalhealth.tsv.gz', sep='\t',
                  encoding='utf-8', compression='gzip', index=False)
    # Call sep_data: also save the different categories in a separate file
    sep_data(df_QOL, create_file_with_groups_path, 'general_health')


def add_income(create_file_with_groups_path, path_directory, config):
    """
    Add income
    """
    # Read file
    df_QOL = pd.read_csv(f'{create_file_with_groups_path}merge_15ormore_household_media_generalhealth.tsv.gz', sep='\t',
                         encoding='utf-8',
                         compression='gzip')
    inter_column = list()
    interesting_column = '_income_'
    df_quest = pd.DataFrame(columns=['project_pseudo_id'])
    # Get the income questions
    for files in os.listdir(path_directory):
        if files.startswith('cov'):
            filenum = files.split('_')[2]
            filenum = str(filenum.replace('t', ''))
            dict_ans = dict()
            df = pd.read_csv(f'{path_directory}{files}', sep=',', encoding='utf-8')
            # Replace none_values with np.nan in df
            none_value = ['"$4"', '"$5"', '"$6"', '"$7"', '$4', '$5', '$6', '$7']
            df[df.isin(none_value)] = np.nan
            for col in df.columns:
                if interesting_column in col:
                    inter_column.append(col)
                    variables_df = pd.read_csv(
                        f"{config['path_questionnaire_enumerations']}{files.replace('results', 'enumerations')}",
                        sep=',', encoding='utf-8')
                    select_var = variables_df[variables_df['variable_name'] == col]
                    for index, row in select_var.iterrows():
                        dict_ans[str(row['enumeration_code'])] = row['enumeration_en']
                    # df[col] = df[col].map(dict_ans)
                    df_quest = pd.merge(df_quest, df[['project_pseudo_id', col]], how='outer', on=['project_pseudo_id'])
    # Update answers of older questions with answers of new questions
    df_quest['income'] = df_quest[['covt08_income_adu_q_2']]
    df_quest['income'].update(df_quest[df_quest['covt10_income_adu_q_2'].notnull()]['covt10_income_adu_q_2'])
    df_quest['income'].update(df_quest[df_quest['covt13_income_adu_q_2'].notnull()]['covt13_income_adu_q_2'])
    # Merge dataframes
    df_QOL = pd.merge(df_QOL, df_quest[['project_pseudo_id', 'income']], on=['project_pseudo_id'], how="left")
    df_QOL.to_csv(f'{create_file_with_groups_path}merge_15ormore_household_media_generalhealth_income.tsv.gz', sep='\t',
                  encoding='utf-8', compression='gzip', index=False)
    # Call sep_data: also save the different categories in a separate file
    sep_data(df_QOL, create_file_with_groups_path, 'income')


def add_educational_level(create_file_with_groups_path, path_directory, config):
    """
    Add education level data
    # General http://wiki-lifelines.web.rug.nl/doku.php?id=education
    """
    # Read file
    df_QOL = pd.read_csv(f'{create_file_with_groups_path}merge_15ormore_household_media_generalhealth_income.tsv.gz',
                         sep='\t', encoding='utf-8',
                         compression='gzip')
    list_columns = list()
    general = 'degree_highest_adu_q_1'
    df_edu = pd.DataFrame(columns=['project_pseudo_id'])

    # Get general health questions
    for file in os.listdir(path_directory):
        if file.startswith('covq'):
            df = pd.read_csv(f"{path_directory}{file}", sep=',', encoding='utf-8')
            # Replace none_values with np.nan in df
            none_value = ['"$4"', '"$5"', '"$6"', '"$7"', '$4', '$5', '$6', '$7']
            df[df.isin(none_value)] = np.nan
            for col in df.columns:
                if general in col and not col.endswith('_a'):
                    df[col] = df[col].astype(str)
                    list_columns.append(f"{file.replace('_results.csv', '')}_{col}")
                    df_edu = pd.merge(df_edu, df[['project_pseudo_id', col]], on=['project_pseudo_id'], how="outer")
                    df_edu = df_edu.rename(columns={col: f"{file.replace('_results.csv', '')}_{col}"})

    # Get general health questions
    # covt22
    covt_22 = '_degree_adu_q_1'
    for i in range(21, 23):
        if i < 10:
            i = f'0{i}'
        df = pd.read_csv(f'{path_directory}covq_q_t{i}_results.csv', sep=',', encoding='utf-8')
        # Replace none_values with np.nan in df
        none_value = ['"$4"', '"$5"', '"$6"', '"$7"', '$4', '$5', '$6', '$7']
        df[df.isin(none_value)] = np.nan
        for col in df.columns:
            if covt_22 in col and ~col.endswith('_a'):
                df[col] = df[col].astype(str)
                list_columns.append(col)
                df_edu = pd.merge(df_edu, df[['project_pseudo_id', col]], on=['project_pseudo_id'], how="outer")
    # Sort list   
    list_columns = sorted(list_columns)
    df_edu['education'] = df_edu[list_columns[0]]
    # Update answers of older questions with answers of new questions
    for i in range(1, len(list_columns)):
        df_edu['education'].update(df_edu[df_edu[list_columns[i]].notnull()][list_columns[i]])
    # Merge dataframes
    df_QOL = pd.merge(df_QOL, df_edu[['project_pseudo_id', 'education']], on=['project_pseudo_id'], how="left")
    df_QOL.to_csv(f'{create_file_with_groups_path}merge_15ormore_household_media_generalhealth_income_education.tsv.gz',
                  sep='\t',
                  encoding='utf-8', compression='gzip', index=False)
    # Call sep_data: also save the different categories in a separate file
    sep_data(df_QOL, create_file_with_groups_path, 'education')


def sep_data(df_QOL, create_file_with_groups_path, cat):
    """
    Also save the different categories in a separate file
    """
    cat_columns = ['project_pseudo_id']
    for col in df_QOL.columns:
        if cat in col:
            cat_columns.append(col)
    df_cat = df_QOL[cat_columns]
    df_cat = df_cat.drop_duplicates().reset_index()
    del df_cat["index"]
    df_cat.to_csv(f'{create_file_with_groups_path}sep_{cat}.tsv.gz', sep='\t',
                  encoding='utf-8', compression='gzip', index=False)


def main():
    config = get_config()
    path_directory = config['path_questionnaire_results']
    create_file_with_groups_path = config['create_file_with_groups']
    question_15_or_more_path = config['question_15_or_more']
    data_QOL = config['data_QOL']

    # print('merge_questdata_with_other')
    merge_questdata_with_other(data_QOL, create_file_with_groups_path, question_15_or_more_path)
    # print('add_media_data')
    add_media_data(path_directory, create_file_with_groups_path)
    # print('add_general_health')
    add_general_health(create_file_with_groups_path, path_directory, config)
    # print('add_income')
    add_income(create_file_with_groups_path, path_directory, config)
    # print('add_educational_level')
    add_educational_level(create_file_with_groups_path, path_directory, config)

    print('DONE: create_file_with_groups.py')


if __name__ == '__main__':
    main()
