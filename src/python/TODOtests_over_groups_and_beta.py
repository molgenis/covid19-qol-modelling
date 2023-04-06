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

def get_data_ready(path_read_QOL, path_save):
    """
    
    """
    # Read files
    df_QOL = pd.read_csv(f'{path_read_QOL}merge_15ormore_household_media_generalhealth_income_education.tsv.gz',
                             sep='\t', encoding='utf-8', compression='gzip')
    df = pd.read_csv(f'{path_read_QOL}num_quest_1_filter.tsv.gz', sep='\t', encoding='utf-8',
                                compression='gzip')
    # Merge files
    df_QOL = pd.merge(df_QOL, df, how='left', on=['project_pseudo_id', 'times_part'])
    # Save file
    df_QOL = pd.read_csv(f'{path_read_QOL}with_beta_without_respons.tsv.gz',
                             sep='\t', encoding='utf-8', compression='gzip')
    # Read file
    df_BFI = pd.read_csv(f'/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/VL/BFI_only_with_sum.tsv.gz',
                             sep='\t', encoding='utf-8', compression='gzip')
    # Remove 'Unnamed: 0'
    df_BFI = df_BFI.iloc[:, 1:]
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
    df_QOL.to_csv(f'{path_read_QOL}merge_no_mini_last.tsv.gz', sep='\t', encoding='utf-8',
                        compression='gzip')
    
    return df_QOL 

def calculate_U(group_a, group_b, name_a, name_b):
    """
    
    Calculate U
    
    """
    #http://users.sussex.ac.uk/~grahamh/RM1web/MannWhitneyHandout%202011.pdf
    a_b = list(group_a) + list(group_b)
    a_b_values = list([0] * len(group_a)) + list([1] * len(group_b))
    df = pd.DataFrame(data = {'beta': a_b, 'group': a_b_values})
    # Step 1
    df['average_rank'] = df['beta'].rank(method='average')
    # Step 2
    T1 = sum(df[df['group'] == 0]['average_rank'])
    # Step 3
    T2 = sum(df[df['group'] == 1]['average_rank'])
    # Step 4
    TX = max(T1, T2)
    # Step 5
    N1 = len(group_a)
    N2 = len(group_b)
    if TX == T1:
        NX = N1
    else:
        NX = N2
    # Step 6
    U = N1 * N2 + NX * (NX + 1) / 2 - TX
    # wikipedia: https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test
    U1 = T1 - (N1 * (N1 + 1)) / 2
    U2 = T2 - (N2 * (N2 + 1)) / 2
    print(f'T1:{T1} - T2:{T2} - TX:{TX}')
    print(f'N1:{N1} - N2:{N2} - NX:{NX}')
    print(f'U:{U}')
    print(f'U1:{U1} - U2:{U2}')
    if U1 > U2:
        print(f'U1 ({name_a}) > U2 ({name_b})')
    else:
        print(f'U1 ({name_a}) < U2 ({name_b})')
    print(f'{name_a} - median:{median(list(group_a))}, U1:{U1}, mean:{mean(list(group_a))}, len: {len(group_a)}')
    print(f'{name_b} - median:{median(list(group_b))}, U2:{U2}, mean:{mean(list(group_b))}, len: {len(group_b)}')
    print()

def calculate_group(df_QOL_select, column_group):
    """
    
    """
    female_df = df_QOL_select[df_QOL_select['gender'] == 'FEMALE']
    male_df = df_QOL_select[df_QOL_select['gender'] != 'FEMALE']
    print(column_group)
    print(f'female: {len(female_df)} - mean_age: {female_df["mean_age"].mean()}')
    print(f'male: {len(male_df)} - mean_age: {male_df["mean_age"].mean()}')

def select_columns(df_QOL, variable, column_group):
    """
    
    """
    df_QOL = df_QOL[df_QOL['mean_age'].notna()]
    df_QOL = df_QOL[df_QOL['gender'].notna()]
    df_QOL_select = df_QOL[['project_pseudo_id', column_group, variable]]  
    df_QOL_select = df_QOL_select.dropna().reset_index(drop=True)
    return df_QOL_select

def spearman_test(df_QOL_select, variable, column_group):
    """
    
    """
    spearman_values = stats.spearmanr(df_QOL_select[column_group], df_QOL_select[variable])
    print(f'-----{column_group} - spearmanr')
    print(spearman_values)
    print(spearman_values.pvalue)

def wilcoxon_U_test(first_group, second_group, column_group):
    """
    
    """
    # wilcoxon_values = stats.ranksums(first_group, second_group)
    wilcoxonU = stats.mannwhitneyu(first_group, second_group)
    print(f'-----{column_group} - mannwhitneyu')
    print(wilcoxonU)
    print(wilcoxonU.pvalue)


def age(df_QOL, variable):
    """
    
    """
    column_group = 'mean_age'
    df_QOL_select = select_columns(df_QOL, variable, column_group)
    spearman_test(df_QOL_select, variable, column_group)


def gender(df_QOL, variable):
    """
    
    """
    column_group = 'gender'
    df_QOL_select = select_columns(df_QOL, variable, column_group)    
    beta_female = df_QOL_select[df_QOL_select[column_group] == 'FEMALE'][variable]
    beta_male = df_QOL_select[df_QOL_select[column_group] != 'FEMALE'][variable]
    
    wilcoxon_U_test(beta_female, beta_male, column_group)

    # calculate_U(beta_female, beta_male, 'Female', 'Male')
    # calculate_group(df_QOL_select[df_QOL_select['gender'] == 'FEMALE'], column_group)
    # calculate_group(df_QOL_select[df_QOL_select['gender'] != 'FEMALE'], column_group)
    

def household(df_QOL, variable):
    """
    
    """
    df_QOL_select = df_QOL[['project_pseudo_id', 'mean_age', 'gender', 'household_status', variable]]
    df_QOL_select = df_QOL_select[df_QOL_select['household_status'].notna()]
    df_QOL_select = df_QOL_select.dropna().reset_index(drop=True)
    
    for cat_household in list(set(df_QOL_select['household_status'])):
        beta_cat = df_QOL_select[df_QOL_select['household_status'] == cat_household][variable]
        beta_other = df_QOL_select[df_QOL_select['household_status'] != cat_household][variable]
        
        wilcoxon_U_test(beta_cat, beta_other, cat_household)

        # calculate_U(beta_cat, beta_other, cat_household, 'other')
        # calculate_group(df_QOL_select[df_QOL_select['household_status'] == cat_household], cat_household)
        # calculate_group(df_QOL_select[df_QOL_select['household_status'] != cat_household], 'other')


def media(df_QOL, variable):
    """
    
    """
    media_cols = [col for col in df_QOL.columns if 'mediacategory' in col]
    
    for cat_media in media_cols:
        df_QOL_select = df_QOL[['project_pseudo_id', 'mean_age', 'gender', cat_media, variable]]
        df_QOL_select = df_QOL_select.dropna().reset_index(drop=True)
        
        beta_False = df_QOL_select[df_QOL_select[cat_media] == False][variable]
        beta_True = df_QOL_select[df_QOL_select[cat_media] != False][variable]

        wilcoxon_U_test(beta_False, beta_True, cat_media)

        # calculate_U(beta_False, beta_True, 'False', 'True')
        # calculate_group(df_QOL_select[df_QOL_select[cat_media] != False], cat_media)
        # calculate_group(df_QOL_select[df_QOL_select[cat_media] == False], 'other')
        

def general_health(df_QOL, variable):
    """
    
    """
    column_group = 'general_health'
    df_QOL_select = select_columns(df_QOL, variable, column_group)
    gen_health = {'poor': 1,'mediocre': 2, 'good': 3, 'very good': 4, 'excellent': 5 }
    df_QOL_select[column_group] = df_QOL_select[column_group].map(gen_health)
    spearman_test(df_QOL_select, variable, column_group)


def education(df_QOL, variable):
    """
    
    """
    column_group = 'education'
    df_QOL_select = select_columns(df_QOL, variable, column_group)
    educat = {1.0:1, 2.0:7, 3.0:10, 4.0:10, 5.0:15, 6.0:13, 7.0:22, 8.0:22, 9.0:np.nan}
    df_QOL_select[column_group] = df_QOL_select[column_group].map(educat)
    df_QOL_select = df_QOL_select.dropna().reset_index(drop=True)
    spearman_test(df_QOL_select, variable, column_group)


def income(df_QOL, variable):
    """
    
    """
    column_group = 'income'
    df_QOL_select = select_columns(df_QOL, variable, column_group)
    
    dict_ans = {"<\u20AC500":1.0, "\u20AC501-\u20AC1000":2.0, "\u20AC1001-\u20AC1500":3.0, "\u20AC1501-\u20AC2000":4.0, 
                "\u20AC2001-\u20AC2500":5.0, "\u20AC2501-\u20AC3000":6.0, "\u20AC3001-\u20AC3500":7.0, 
                "\u20AC3501-\u20AC4000":8.0, "\u20AC4001-\u20AC4500":9.0, "\u20AC4501-\u20AC5000":10.0, 
                "\u20AC5001-\u20AC7500":11.0, ">\u20AC7500":12.0, "I prefer not to say":13.0}
    df_QOL_select[column_group] = df_QOL_select[column_group].map(dict_ans)
    df_QOL_select[column_group].replace(13, np.NaN, inplace=True)
    df_QOL_select = df_QOL_select.dropna().reset_index(drop=True)
    
    spearman_test(df_QOL_select, variable, column_group)
    

def mini_dep(df_QOL, variable):
    """
    
    """
    column_group_dep = 'major_depressive_episode'
    
    df_QOL_select_dep = df_QOL[['project_pseudo_id', 'mean_age', 'gender', column_group_dep, variable]]
    # df_QOL_select_dep = df_QOL_select_dep.dropna().reset_index(drop=True)
    df_QOL_select_dep = df_QOL_select_dep[df_QOL_select_dep[variable].notna()]
    df_QOL_select_dep[column_group_dep] = df_QOL_select_dep[column_group_dep].fillna(0)
    beta_depression = df_QOL_select_dep[df_QOL_select_dep[column_group_dep] == 1][variable]
    beta_notdep = df_QOL_select_dep[df_QOL_select_dep[column_group_dep] != 1][variable]

    wilcoxon_U_test(beta_depression, beta_notdep, column_group_dep)

    # calculate_U(beta_depression, beta_notdep, 'dep', 'NOT dep')
    # calculate_group(df_QOL_select_dep[df_QOL_select_dep[column_group_dep] == 1], column_group_dep)
    # calculate_group(df_QOL_select_dep[df_QOL_select_dep[column_group_dep] != 1], 'Not dep')



def mini_anx(df_QOL, variable):
    """
    
    """
    column_group_anx = 'generalized_anxiety_disorder'
    df_QOL_select_anx = df_QOL[['project_pseudo_id', 'mean_age', 'gender', column_group_anx, variable]]
    # df_QOL_select_anx = df_QOL_select_anx.dropna().reset_index(drop=True)
    df_QOL_select_anx = df_QOL_select_anx[df_QOL_select_anx[variable].notna()]
    df_QOL_select_anx[column_group_anx] = df_QOL_select_anx[column_group_anx].fillna(0)
    beta_anx = df_QOL_select_anx[df_QOL_select_anx[column_group_anx] == 1][variable]
    beta_notanx = df_QOL_select_anx[df_QOL_select_anx[column_group_anx] != 1][variable]

    wilcoxon_U_test(beta_anx, beta_notanx, column_group_anx)

    # calculate_U(beta_anx, beta_notanx, 'anx', 'NOT anx')
    # calculate_group(df_QOL_select_anx[df_QOL_select_anx[column_group_anx] == 1], column_group_anx)
    # calculate_group(df_QOL_select_anx[df_QOL_select_anx[column_group_anx] != 1], 'Not anx')


def BFI(df_QOL, variable):
    """
    
    """
    BFI_col = [col for col in df_QOL.columns if '_sum' in col]
    for col in BFI_col:
        if 'before_2a' not in col and 'between' not in col:
            df_QOL_select = df_QOL[['project_pseudo_id', col, variable]]
            df_QOL_select = df_QOL_select.dropna().reset_index(drop=True)
            spearman_test(df_QOL_select, variable, col)
            

def resilience(df_QOL, variable):
    """
    
    """
    column_group = 'resilience_median'
    df_QOL_select = select_columns(df_QOL, variable, column_group)
    spearman_test(df_QOL_select, variable, column_group)
    


def beta(df, path_read_QOL):
    """
    
    """
    # beta
    df = df[df['beta'].notna()]
    df_sort = df.sort_values(by=['beta'])
    beta_participant = df_sort[['project_pseudo_id', 'beta']].drop_duplicates()
    n = 10
    head_n = beta_participant.head(int(len(beta_participant)*(n/100)))
    tail_n = beta_participant.tail(int(len(beta_participant)*(n/100)))

    head_nn = beta_participant[beta_participant['project_pseudo_id'].isin(list(head_n['project_pseudo_id']))]
    head_nn['beta_type'] = 'bottom'
    tail_nn = beta_participant[beta_participant['project_pseudo_id'].isin(list(tail_n['project_pseudo_id']))]
    tail_nn['beta_type'] = 'top'
    head_tail = pd.merge(head_nn[['project_pseudo_id', 'beta', 'beta_type']], tail_nn[['project_pseudo_id', 'beta', 'beta_type']], how='outer', on=['project_pseudo_id', 'beta', 'beta_type'])
    print(head_tail)

    
    beta_participant['beta_abs'] = beta_participant['beta'].abs()
    df_sort_abs = beta_participant.sort_values(by=['beta_abs'])
    print(df_sort_abs)
    beta_participant_abs = df_sort_abs[['project_pseudo_id', 'beta_abs']].drop_duplicates()
    head_n_abs = beta_participant_abs.head(int(len(beta_participant_abs)*(n/100)))
    print(len(head_n_abs))
    print(head_n_abs)
    print('-----------')
    print(beta_participant)
    head_df_abs = beta_participant[beta_participant['project_pseudo_id'].isin(list(head_n_abs['project_pseudo_id']))]
    head_df_abs['beta_type'] = 'around_zero'
    print(head_n_abs)

    # head_df_abs.to_csv(f'{path_read_QOL}head_{n}_beta_abs.tsv', sep='\t', encoding='utf-8',
    #                     index=False)
    head_tail_null_df = pd.merge(head_tail[['project_pseudo_id', 'beta', 'beta_type']], head_df_abs[['project_pseudo_id', 'beta', 'beta_type']], how='outer', on=['project_pseudo_id', 'beta', 'beta_type'])
    head_tail_null_df.to_csv(f'{path_read_QOL}head_tail_null_{n}_beta_abs.tsv', sep='\t', encoding='utf-8',
                        index=False)
    print(head_tail_null_df)
    print(f'========= {len(set(head_tail_null_df["project_pseudo_id"]))}')
    print(set(head_tail_null_df['beta_type']))
    print(list(head_tail_null_df.columns))
    print(int(len(beta_participant)*(n/100)))


    


def main():
    print('start')
    config = get_config()
    my_folder = config['my_folder']
    path_read_QOL = config['path_read_QOL']
    path_save = f'{path_read_QOL}with_just_mean/'
    filter_num = 15
    variable = 'beta'
    # df_QOL = get_data_ready(path_read_QOL, path_save, filter_num)
    
    df_QOL = pd.read_csv(f'{path_read_QOL}merge_no_mini_last.tsv.gz' , sep='\t', encoding='utf-8', compression='gzip') #merge_no_mini_filter merge_no_mini_NOfilter merge_no_mini_last
    veerkracht = pd.read_csv(f"{path_read_QOL}df/veerkracht.tsv.gz" , sep='\t', encoding='utf-8', compression='gzip')
    mini = pd.read_csv(f"{path_read_QOL}df/between_before_mini.tsv.gz" , sep='\t', encoding='utf-8', compression='gzip')
    df_QOL = pd.merge(df_QOL, veerkracht, on=['project_pseudo_id'], how='outer')
    df_QOL = pd.merge(df_QOL, mini, on=['project_pseudo_id'], how='outer')
    df_QOL = df_QOL[df_QOL['age'].notna()]
    df_QOL = df_QOL[df_QOL['gender'].notna()]
    df_QOL['major_depressive_episode'] = df_QOL['major_depressive_episode'].fillna(0)
    df_QOL['generalized_anxiety_disorder'] = df_QOL['generalized_anxiety_disorder'].fillna(0)

    age_mean = df_QOL.groupby('project_pseudo_id')['age'].mean().reset_index()
    age_mean = age_mean.rename(columns={age_mean.columns[1]: 'mean_age'})
    print(age_mean)
    df_QOL = pd.merge(df_QOL, age_mean, on=['project_pseudo_id'], how="left")
    
   
    df_beta_type = pd.read_csv(f'{path_read_QOL}head_tail_null_10_beta_abs.tsv' , sep='\t', encoding='utf-8')
    df_QOL = pd.merge(df_QOL, df_beta_type[['project_pseudo_id', 'beta_type']], how='left', on=['project_pseudo_id'])

    df_QOL_select = df_QOL[['project_pseudo_id', 'responsedate', 'qualityoflife',
                    'gender', 'age', 'mean_age', 'household_status', 'general_health',
                    'income', 'mediacategory_media', 'mediacategory_health_authorities',
                    'mediacategory_social_media', 'mediacategory_family_and_friends',
                    'mediacategory_other', 'education', 'n_sum', 'e_sum', 'o_sum', 'a_sum',
                    'c_sum', 'times_part', 'beta', 'resilience_mean', 'resilience_median',
                    'major_depressive_episode', 'generalized_anxiety_disorder', 'num_quest',
                    'beta_type']]
    
    df_QOL_select.to_csv(f'{path_read_QOL}QOL_selected_columns_withbetatypes.tsv.gz', sep='\t', encoding='utf-8',
                        compression='gzip', index=False)
    
    df_QOL_select2 = df_QOL_select.drop(['responsedate', 'qualityoflife', 'age', 'num_quest'], axis=1)
    df_QOL_select2.drop_duplicates(inplace=True)
    df_QOL_select2.to_csv(f'{path_read_QOL}QOL_selected_columns_withbetatypes_noageresponsequalnumquest.tsv.gz', sep='\t', encoding='utf-8',
                        compression='gzip', index=False)

    # df_QOL_select = pd.read_csv(f'{path_read_QOL}QOL_selected_columns_withbetatypes.tsv.gz' , sep='\t', encoding='utf-8', compression='gzip') #merge_no_mini_filter merge_no_mini_NOfilter merge_no_mini_last    
    
    age(df_QOL_select, variable)
    df_QOL_select = df_QOL_select.drop(['responsedate', 'qualityoflife', 'age', 'num_quest'], axis=1)
    df_QOL_select.drop_duplicates(inplace=True)
    gender(df_QOL_select, variable)
    household(df_QOL_select, variable)
    media(df_QOL_select, variable)
    general_health(df_QOL_select, variable)
    education(df_QOL_select, variable)
    income(df_QOL_select, variable)
    mini_dep(df_QOL, variable)
    mini_anx(df_QOL, variable)
    BFI(df_QOL_select, variable)
    resilience(df_QOL_select, variable)


    # variable = 'beta_abs'
    # df_QOL_filter['beta_abs'] = df_QOL_filter['beta'].abs()
    # print(df_QOL_filter)
    # # # print(df_QOL_filter)
    # age(df_QOL_filter, variable)
    # # ## df_QOL_filter.drop('age', axis=1, inplace=True)
    # # ## df_QOL_filter.drop_duplicates(inplace=True)
    # gender(df_QOL_filter, variable)
    # household(df_QOL_filter, variable)
    # media(df_QOL_filter, variable)
    # general_health(df_QOL_filter, variable)
    # education(df_QOL_filter, variable)
    # income(df_QOL_filter, variable)
    # mini_ana(df_QOL_filter, variable)
    # BFI(df_QOL_filter, variable)
    # resilience(df_QOL_filter, variable)
    
    print('DONE')


if __name__ == '__main__':
    main()

