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
import sys

sys.path.append(
    '/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/covid19-qol-modelling/src/python')
from config import get_config
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import warnings

warnings.filterwarnings('ignore')


def make_df_beta(question_15_or_more_path, calculate_beta_path, head_top_null_path):
    """
    Merge two files
    """
    # Read files
    QOL = pd.read_csv(f'{question_15_or_more_path}num_quest_1_filter.tsv.gz', sep='\t',
                      encoding='utf-8', compression='gzip')
    QOL = QOL[['project_pseudo_id', 'times_part', 'responsedate', 'qualityoflife']]
    beta = pd.read_csv(f'{calculate_beta_path}QOL_covariance_correlation_beta.tsv.gz', sep='\t',
                       encoding='utf-8', compression='gzip')
    # Merge files
    QOL_beta = pd.merge(QOL[['project_pseudo_id', 'responsedate', 'qualityoflife']],
                        beta[['project_pseudo_id', 'beta']], how='outer', on=['project_pseudo_id'])
    QOL_beta.to_csv(f'{head_top_null_path}QOL_beta.tsv.gz', sep='\t', encoding='utf-8',
                    compression='gzip')
    return QOL_beta


def read_df_beta(head_top_null_path, df):
    """

    """
    # Read file
    if df.empty:
        df = pd.read_csv(f'{head_top_null_path}QOL_beta.tsv.gz', sep='\t',
                         encoding='utf-8', compression='gzip')
    # Beta not nan
    df = df[df['beta'].notna()]
    # Sort values
    df_sort = df.sort_values(by=['beta'])
    beta_participant = df_sort[['project_pseudo_id', 'beta']].drop_duplicates()
    # ...%
    n = 10
    # How many we want (top or bottom ...%)
    head_n = beta_participant.head(int(len(beta_participant) * (n / 100)))
    tail_n = beta_participant.tail(int(len(beta_participant) * (n / 100)))
    # Select the head and tail participants
    head_nn = df_sort[df_sort['project_pseudo_id'].isin(list(head_n['project_pseudo_id']))]
    head_nn['beta_type'] = 'bottom'
    tail_nn = df_sort[df_sort['project_pseudo_id'].isin(list(tail_n['project_pseudo_id']))]
    tail_nn['beta_type'] = 'top'
    # Merge head and tails ...%
    head_tail = pd.merge(head_nn[['project_pseudo_id', 'responsedate', 'qualityoflife', 'beta', 'beta_type']],
                         tail_nn[['project_pseudo_id', 'responsedate', 'qualityoflife', 'beta', 'beta_type']],
                         how='outer', on=['project_pseudo_id', 'responsedate', 'qualityoflife', 'beta', 'beta_type'])
    # Take the absolute value of the beta
    df_sort['beta_abs'] = df_sort['beta'].abs()
    df_sort_abs = df_sort.sort_values(by=['beta_abs'])
    beta_participant_abs = df_sort_abs[['project_pseudo_id', 'beta_abs']].drop_duplicates()
    head_n_abs = beta_participant_abs.head(int(len(beta_participant_abs) * (n / 100)))
    head_df_abs = df_sort[df_sort['project_pseudo_id'].isin(list(head_n_abs['project_pseudo_id']))]
    head_df_abs['beta_type'] = 'around_zero'
    # Merge
    head_tail_null_df = pd.merge(head_tail[['project_pseudo_id', 'responsedate', 'qualityoflife', 'beta', 'beta_type']],
                                 head_df_abs[
                                     ['project_pseudo_id', 'responsedate', 'qualityoflife', 'beta', 'beta_type']],
                                 how='outer',
                                 on=['project_pseudo_id', 'responsedate', 'qualityoflife', 'beta', 'beta_type'])
    # Write files
    head_nn.to_csv(f'{head_top_null_path}head_{n}_beta.tsv.gz', sep='\t', encoding='utf-8',
                   compression='gzip', index=False)
    tail_nn.to_csv(f'{head_top_null_path}tail_{n}_beta.tsv.gz', sep='\t', encoding='utf-8',
                   compression='gzip', index=False)
    head_tail.to_csv(f'{head_top_null_path}head_tail_{n}_beta.tsv.gz', sep='\t', encoding='utf-8',
                     compression='gzip', index=False)
    head_df_abs.to_csv(f'{head_top_null_path}head_{n}_beta_abs.tsv.gz', sep='\t', encoding='utf-8',
                       compression='gzip', index=False)
    head_tail_null_df.to_csv(f'{head_top_null_path}head_tail_null_{n}_beta_abs.tsv.gz', sep='\t', encoding='utf-8',
                             compression='gzip', index=False)


def main():
    config = get_config()
    head_top_null_path = config['head_top_null']
    question_15_or_more_path = config['question_15_or_more']
    calculate_beta_path = config['calculate_beta']
    QOL_beta = pd.DataFrame()
    QOL_beta = make_df_beta(question_15_or_more_path, calculate_beta_path, head_top_null_path)
    read_df_beta(head_top_null_path, QOL_beta)
    print('DONE: head_top_null_beta.py')


if __name__ == '__main__':
    main()
