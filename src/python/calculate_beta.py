#!/usr/bin/env python3

# Imports
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import collections
from collections import Counter
import math

import sys
sys.path.append(
    '/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/covid19-qol-modelling/src/python')
from config import get_config

def calculate_covariance_QOL(path_save, make_df_id):
    """
    Calculates the covariance in steps.
    df_id : dataframe with responsdate, project_pseudo_id and quality of life
    df_q : dataframe with responsdate and mean quality of life
    """
    # Read file: df_id
    df_id = pd.read_csv(f'{make_df_id}df_id.tsv.gz', sep='\t', encoding='utf-8', compression='gzip').iloc[: , 1:]
    df_id['responsedate'] = pd.to_datetime(df_id['responsedate'])
    df_id = df_id.astype({'qualityoflife': 'int64'})
    # Read file: df_q
    df_q = pd.read_csv(f'{make_df_id}df_q.tsv.gz', sep='\t', encoding='utf-8', compression='gzip').iloc[: , 1:]
    # Rename column
    df_q.rename(columns={'qualityoflife': 'qualityoflife_mean'}, inplace=True)
    df_q = df_q.astype({'qualityoflife_mean': 'float64'})
    df_q['responsedate'] = pd.to_datetime(df_q['responsedate'])
       
    # Concat dataframes
    df_QOL = pd.merge(df_q, df_id, on=['responsedate'])
    # Calculate the x-mean (mean per person QOL)
    x_mean = df_QOL.groupby('project_pseudo_id')['qualityoflife'].mean().reset_index().rename(
        columns={'qualityoflife': 'x_mean'})
    # Merge x_mean to df_QOL
    df_QOL = pd.merge(df_QOL, x_mean, on=['project_pseudo_id'])
    # Calculate the y-mean (mean of the QOL mean)
    y_mean = df_QOL.groupby('project_pseudo_id')['qualityoflife_mean'].mean().reset_index().rename(
        columns={'qualityoflife_mean': 'y_mean'})
    # Merge y_mean to df_QOL
    df_QOL = pd.merge(df_QOL, y_mean, on=['project_pseudo_id'])
    # See how often people have a response date
    times_person = df_QOL.groupby('project_pseudo_id').size().reset_index().rename(columns={0: 'times'})
    # Calculate (x - x_head) = (QOL per day of a person - average QOL per person) = (qualityoflife - x_mean)
    df_QOL['x_diff'] = df_QOL['qualityoflife'] - df_QOL['x_mean']
    # Calculate (y - y_head) = (Average QOL of the average QOL score per day - average QOL in a day) =
    # (qualityoflife_mean - y_mean)
    df_QOL['y_diff'] = df_QOL['qualityoflife_mean'] - df_QOL['y_mean']
    # Calculate the (x - x_head)*(y - y_head)
    df_QOL['x_diff_TIMES_y_diff'] = df_QOL['x_diff'] * df_QOL['y_diff']
    # Calculate the numerator sum((x - x_head)*(y - y_head))
    numerator_covariance = df_QOL.groupby('project_pseudo_id')['x_diff_TIMES_y_diff'].sum().reset_index()
    # Merge times_person and numerator_covariance
    covariance = pd.merge(times_person, numerator_covariance, on=['project_pseudo_id'])
    # Calculate the covariance numerator / n-1
    covariance['covariance_value'] = covariance['x_diff_TIMES_y_diff'] / (covariance['times'] - 1)
    # Write df to csv file
    df_QOL.to_csv(f'{path_save}df_QOL.tsv.gz', sep='\t', encoding='utf-8', compression='gzip')
    # Save dataframe
    covariance.to_csv(f'{path_save}QOL_covariance.tsv.gz', sep='\t', encoding='utf-8', compression='gzip')
    return df_QOL, covariance


def calculate_cor_and_beta(path_save, df_QOL, covariance_df):
    """
    
    """
    # # Read files
    # df_QOL = pd.read_csv(f'{path_save}df_QOL.tsv.gz', sep='\t', encoding='utf-8', compression='gzip')
    # covariance_df = pd.read_csv(f'{path_save}QOL_covariance.tsv.gz', sep='\t', encoding='utf-8', compression='gzip')
    
    # # Remove 'Unnamed: 0'
    # covariance_df = covariance_df.iloc[:, 1:]
    # Fill NaN with 0. This is because when the numerator is 0 the covariance is also 0.
    covariance_df['covariance_value'] = covariance_df['covariance_value'].fillna(0)
    # Select only the people with more then 15 questionnaires
    covariance_df = covariance_df[covariance_df['times'] >= 15]
    # See how often people have a response date
    times_person = df_QOL.groupby('project_pseudo_id').size().reset_index().rename(columns={0: 'times'})
    # Calculate (x - x_head) = (QOL per day of a person - average QOL per person) = (qualityoflife - x_mean)
    df_QOL['x_diff'] = df_QOL['qualityoflife'] - df_QOL['x_mean']
    # Calculate (x - x_head)^2
    df_QOL['x_diff_2'] = df_QOL['x_diff'] ** 2
    # Calculate (y - y_head) = (Average QOL of the average QOL score per day - average QOL in a day) =
    # (qualityoflife_mean - y_mean)
    df_QOL['y_diff'] = df_QOL['qualityoflife_mean'] - df_QOL['y_mean']
    # Calculate (y - y_head)^2
    df_QOL['y_diff_2'] = df_QOL['y_diff'] ** 2
    # Calculate sum((x - x_head)^2)
    x_dif_2_sum = df_QOL.groupby('project_pseudo_id')['x_diff_2'].sum().reset_index()
    # Calculate sum((y - y_head)^2)
    y_dif_2_sum = df_QOL.groupby('project_pseudo_id')['y_diff_2'].sum().reset_index()
    # Merge dataframes
    calculate_x = pd.merge(times_person, x_dif_2_sum, on=['project_pseudo_id'])
    calculate_y = pd.merge(times_person, y_dif_2_sum, on=['project_pseudo_id'])
    # Calculate variance
    # Example: sum((x - x_head)^2) / (n-1)
    calculate_x['variance_x'] = calculate_x['x_diff_2'] / (calculate_x['times'] - 1)
    calculate_y['variance_y'] = calculate_y['y_diff_2'] / (calculate_y['times'] - 1)
    # Calculate sd
    # Example: sqrt( sum((x - x_head)^2) / (n-1) )
    calculate_x['sd_x'] = np.sqrt((calculate_x['variance_x']))
    calculate_y['sd_y'] = np.sqrt((calculate_y['variance_y']))
    # Merge dataframes
    calculate_x_y = pd.merge(calculate_x[['project_pseudo_id', 'sd_x', 'x_diff_2', 'variance_x']],
                             calculate_y[['project_pseudo_id', 'sd_y', 'y_diff_2', 'variance_y']],
                             on=['project_pseudo_id'], how='outer')
    
    # Merge dataframes
    merge_sd_cov = pd.merge(covariance_df, calculate_x_y, on=['project_pseudo_id'], how='outer')
    # Filter on non Nan values
    merge_sd_cov = merge_sd_cov[merge_sd_cov['covariance_value'].notna()]
    # Calculate the correlation
    # Example: cov(x,y) / (sd(x) * sd(y))
    merge_sd_cov['correlation'] = merge_sd_cov['covariance_value'] / (merge_sd_cov['sd_x'] * merge_sd_cov['sd_y'])
    # Save dataframe
    merge_sd_cov.to_csv(f'{path_save}QOL_covariance_correlation.tsv.gz', sep='\t', encoding='utf-8', compression='gzip')
    # Calculate beta
    # Example: sum((x - x_head)*(y - y_head)) / sum((y - y_head)^2)
    merge_sd_cov['beta'] = merge_sd_cov['x_diff_TIMES_y_diff'] / merge_sd_cov['y_diff_2']
    # Save dataframe
    merge_sd_cov.to_csv(f'{path_save}QOL_covariance_correlation_beta.tsv.gz', sep='\t', encoding='utf-8',
                        compression='gzip')



def main():
    # Call get_config
    config = get_config()
    # Path to the folder containing the results of each questionnaire.
    path_directory = config['path_questionnaire_results']
    make_df_id = config['make_df_id']
    # Call different functions
    path_save = config['calculate_beta']
    df_QOL, covariance = calculate_covariance_QOL(path_save, make_df_id)
    calculate_cor_and_beta(path_save, df_QOL, covariance)
    print('DONE')


if __name__ == '__main__':
    main()

