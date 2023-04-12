# -*- coding: utf-8 -*-
#!/usr/bin/env python3

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
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import scipy.stats as stats
from statsmodels.stats.weightstats import ttest_ind
from pandas.api.types import CategoricalDtype
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statistics import median, mean, stdev
from collections import Counter

def get_data_ready(path_read_QOL, tests_over_groups_and_beta_path, calculate_beta_path, create_file_with_groups_path, question_15_or_more_path, BFI_path):
    """
    
    """
    # Read files
    df_QOL = pd.read_csv(f'{create_file_with_groups_path}merge_15ormore_household_media_generalhealth_income_education.tsv.gz',
                             sep='\t', encoding='utf-8', compression='gzip')
    df = pd.read_csv(f'{question_15_or_more_path}num_quest_1_filter.tsv.gz', sep='\t', encoding='utf-8',
                                compression='gzip')
    # Merge files
    df_QOL = pd.merge(df_QOL, df, how='left', on=['project_pseudo_id', 'times_part'])
    print(df_QOL)
    # # Save file
    # df_QOL = pd.read_csv(f'{path_read_QOL}with_beta_without_respons.tsv.gz',
    #                          sep='\t', encoding='utf-8', compression='gzip')
    # Read file
    df_BFI = pd.read_csv(f'{BFI_path}BFI_only_with_sum.tsv.gz',
                             sep='\t', encoding='utf-8', compression='gzip')
    sum_cols = [col for col in df_BFI.columns if 'sum' in col]
    df_QOL = pd.merge(df_QOL, df_BFI[['project_pseudo_id'] + sum_cols], on=['project_pseudo_id'], how='left')
    # Read file
    covariance_df = pd.read_csv(f'{calculate_beta_path}QOL_covariance_correlation_beta.tsv.gz', sep='\t', encoding='utf-8',
                                compression='gzip')
    # Remove 'Unnamed: 0'
    covariance_df = covariance_df.iloc[:, 1:]
    # Fill NaN with 0. This is because when the numerator is 0 the covariance is also 0.
    covariance_df['covariance_value'] = covariance_df['covariance_value'].fillna(0)    
    df_QOL = pd.merge(df_QOL, covariance_df[['project_pseudo_id', 'beta']], on=['project_pseudo_id'], how='left')
    # Save file
    df_QOL.to_csv(f'{tests_over_groups_and_beta_path}merge_no_mini_last.tsv.gz', sep='\t', encoding='utf-8',
                        compression='gzip', index=False)
    
    return df_QOL 


    


def main():
    print('start')
    config = get_config()
    tests_over_groups_and_beta_path = config['tests_over_groups_and_beta']
    question_15_or_more_path = config['question_15_or_more']
    BFI_path = config['BFI']
    create_file_with_groups_path = config['create_file_with_groups']
    calculate_beta_path = config['calculate_beta']
    path_read_QOL = config['path_read_QOL']
    
    variable = 'beta'
    df_QOL = get_data_ready(path_read_QOL, tests_over_groups_and_beta_path, calculate_beta_path, create_file_with_groups_path, question_15_or_more_path, BFI_path)
    
    # df_QOL = pd.read_csv(f'{path_read_QOL}merge_no_mini_last.tsv.gz' , sep='\t', encoding='utf-8', compression='gzip') #merge_no_mini_filter merge_no_mini_NOfilter merge_no_mini_last
    # veerkracht = pd.read_csv(f"{path_read_QOL}df/veerkracht.tsv.gz" , sep='\t', encoding='utf-8', compression='gzip')
    # mini = pd.read_csv(f"{path_read_QOL}df/between_before_mini.tsv.gz" , sep='\t', encoding='utf-8', compression='gzip')
    # df_QOL = pd.merge(df_QOL, veerkracht, on=['project_pseudo_id'], how='outer')
    # df_QOL = pd.merge(df_QOL, mini, on=['project_pseudo_id'], how='outer')
    # df_QOL = df_QOL[df_QOL['age'].notna()]
    # df_QOL = df_QOL[df_QOL['gender'].notna()]
    # df_QOL['major_depressive_episode'] = df_QOL['major_depressive_episode'].fillna(0)
    # df_QOL['generalized_anxiety_disorder'] = df_QOL['generalized_anxiety_disorder'].fillna(0)

    # age_mean = df_QOL.groupby('project_pseudo_id')['age'].mean().reset_index()
    # age_mean = age_mean.rename(columns={age_mean.columns[1]: 'mean_age'})
    # print(age_mean)
    # df_QOL = pd.merge(df_QOL, age_mean, on=['project_pseudo_id'], how="left")
    
   
    # df_beta_type = pd.read_csv(f'{path_read_QOL}head_tail_null_10_beta_abs.tsv' , sep='\t', encoding='utf-8')
    # df_QOL = pd.merge(df_QOL, df_beta_type[['project_pseudo_id', 'beta_type']], how='left', on=['project_pseudo_id'])

    # df_QOL_select = df_QOL[['project_pseudo_id', 'responsedate', 'qualityoflife',
    #                 'gender', 'age', 'mean_age', 'household_status', 'general_health',
    #                 'income', 'mediacategory_media', 'mediacategory_health_authorities',
    #                 'mediacategory_social_media', 'mediacategory_family_and_friends',
    #                 'mediacategory_other', 'education', 'n_sum', 'e_sum', 'o_sum', 'a_sum',
    #                 'c_sum', 'times_part', 'beta', 'resilience_mean', 'resilience_median',
    #                 'major_depressive_episode', 'generalized_anxiety_disorder', 'num_quest',
    #                 'beta_type']]
    
    # df_QOL_select.to_csv(f'{tests_over_groups_and_beta_path}QOL_selected_columns_withbetatypes.tsv.gz', sep='\t', encoding='utf-8',
    #                     compression='gzip', index=False)
    
    # df_QOL_select2 = df_QOL_select.drop(['responsedate', 'qualityoflife', 'age', 'num_quest'], axis=1)
    # df_QOL_select2.drop_duplicates(inplace=True)
    # df_QOL_select2.to_csv(f'{tests_over_groups_and_beta_path}QOL_selected_columns_withbetatypes_noageresponsequalnumquest.tsv.gz', sep='\t', encoding='utf-8',
    #                     compression='gzip', index=False)

    
    
    print('DONE')


if __name__ == '__main__':
    main()

