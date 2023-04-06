#!/usr/bin/env python3

# Imports
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(
    '/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/COVID_Anne')
from config import get_config
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def calculating_depressive(path_myfolder, mini_df):
    # 1 = yes, 2 = no
    # MAJOR DEPRESSIVE EPISODE
    # print('bezig met maken van lijsten/sets')
    mini_dep_col = list()
    unique_mini = set()
    number_quest = set()
    for col in mini_df.columns:
        if 'minia' in col:
            mini_df[col] = mini_df[col].astype(str).replace('2.0', 0).replace('1.0', 1).replace('nan', np.nan)
            mini_dep_col.append(col)
            unique_mini.add(col.split('_')[1])
            number_quest.add(f"{col.split('_')[0]}_")
    
    sorted_quest = sorted(number_quest)

    # select_columns

    # print('bezig met mini vragen')
    for quest in sorted_quest:
        matching_quest = [s for s in mini_dep_col if quest in s]
        gradation = [s for s in matching_quest if 'minia3' in s] + ['project_pseudo_id']
        symtomes = [s for s in matching_quest if 'minia3' not in s] + ['project_pseudo_id']
        
        mini_df[f'{quest}sum_minia_3'] = mini_df.loc[:,gradation].sum(axis=1) #df_gradation[gradation[0]] + df_gradation[gradation[1]] #df_gradation.sum(axis=1)
        mini_df[f'{quest}sum_minia_1_2'] = mini_df.loc[:,symtomes].sum(axis=1)
        mini_df[f'{quest}sum_minia_all'] = mini_df.loc[:,[f'{quest}sum_minia_1_2', f'{quest}sum_minia_3']].sum(axis=1)

        """
        pagina 5
        Wanneer het 1 is de sum of hoger heeft deze gene symptomen van depressie
        Vervolgens kan er gekeken worden werke graad door vragen 3
        Wanneer er van vraag 1, 2 en alle vragen van 3 er 5 of meer met ja zijn beantwoord kan er MAJOR DEPRESSIVE
        EPISODE, CURRENT
        """
    # print(mini_df)
    # print(list(mini_df.columns))
    # mini_df.to_csv(f'{path_myfolder}merge_for_model_questdata_wellbeing_household2_media_generalhealth2_PRS_income_education_mini_miniadddep.tsv.gz', sep='\t',
    #                      encoding='utf-8', compression='gzip', index=False)

    cat_columns = ['project_pseudo_id']
    for col in mini_df.columns:
        if 'minia' in col:
            cat_columns.append(col)
    df_cat = mini_df[cat_columns]
    df_cat = df_cat.drop_duplicates().reset_index()
    del df_cat["index"]
    df_cat.to_csv(f'{path_myfolder}df/sep_mini_depressive.tsv.gz', sep='\t',
                         encoding='utf-8', compression='gzip', index=False)
    return df_cat


def calculating_anxiety(path_myfolder, mini_df, set_3c, set_3d, set_3f):
    print('--------')
    # 1 = yes, 2 = no
    # GENERALIZED ANXIETY DISORDER
    mini_anx_col = list()
    unique_mini = set()
    number_quest = set()
    for col in mini_df.columns:
        if 'minio' in col:
            mini_df[col] = mini_df[col].astype(str).replace('2.0', 0).replace('1.0', 1).replace('nan', np.nan)
            mini_anx_col.append(col)
            unique_mini.add(col.split('_')[1])
            number_quest.add(f"{col.split('_')[0]}_")
        if 'minia' in col:
            mini_df[col] = mini_df[col].astype(str).replace('2.0', 0).replace('1.0', 1).replace('nan', np.nan)
    
    sorted_quest = sorted(number_quest)
    sorted_mini = sorted(unique_mini)
    sorted_all_mini = sorted(mini_anx_col + set_3c + set_3d + set_3f)

    print(mini_anx_col)
    print(sorted_all_mini)

    print(list(mini_df.columns))

    for quest in sorted_quest:
        print(quest)
        matching_quest = [s for s in sorted_all_mini if quest in s]
        gradation = [s for s in matching_quest if 'minio3' in s] + ['project_pseudo_id']
        symtomes_1 = [s for s in matching_quest if 'minio1' in s] + ['project_pseudo_id']
        symtomes_2 = [s for s in matching_quest if 'minio2' in s] + ['project_pseudo_id']
        # symtomes_1_2 = symtomes_1 + symtomes_2
        print('------------------')
        print(symtomes_1)

        minio3c = list(filter(lambda x: quest in x, set_3c))
        minio3d = list(filter(lambda x: quest in x, set_3d))
        minio3f = list(filter(lambda x: quest in x, set_3f))
        
        mini_df[f'{quest}sum_minio_1'] = mini_df.loc[:,symtomes_1].sum(axis=1)
        mini_df[f'{quest}sum_minio_2'] = mini_df.loc[:,symtomes_2].sum(axis=1)
        # mini_df[f'{quest}_sum_minio_1_2'] = mini_df.loc[:,symtomes_1_2].sum(axis=1)
        mini_df[f'{quest}all_minio3c'] = mini_df.loc[:,minio3c].sum(axis=1)
        mini_df[f'{quest}minio3c'] = mini_df.loc[:,minio3c].sum(axis=1)
        mini_df[f'{quest}minio3c'][mini_df[f'{quest}minio3c'] > 1] = 1
        mini_df[f'{quest}minio3d'] = mini_df.loc[:,minio3d].sum(axis=1)
        mini_df[f'{quest}minio3f'] = mini_df.loc[:,minio3f].sum(axis=1)
        columns_03 = gradation + [f'{quest}minio3c', f'{quest}minio3d', f'{quest}minio3f']
        mini_df[f'{quest}sum_minio_3'] = mini_df.loc[:,columns_03].sum(axis=1)
        mini_df[f'{quest}sum_minio_all'] = mini_df.loc[:,[f'{quest}sum_minio_1', f'{quest}sum_minio_2', f'{quest}sum_minio_3']].sum(axis=1)
        # print(mini_df)

        # mini_df[f'{quest}_sum_minia_all'] = mini_df.loc[:,[f'{quest}_sum_minia_1_2', f'{quest}_sum_minia_3']].sum(axis=1)
    """
    zie pagina 24
    Wanneer het 1 is de sum of hoger heeft deze gene symptomen van depressie
    Vervolgens kan er gekeken worden werke graad door vragen 3
    Wanneer er van vraag 1, 2 en alle vragen van 3 er 5 of meer met ja zijn beantwoord kan er MAJOR DEPRESSIVE
    EPISODE, CURRENT
    """

    cat_columns = ['project_pseudo_id']
    for col in mini_df.columns:
        if 'minio' in col:
            cat_columns.append(col)
    df_cat = mini_df[cat_columns]
    df_cat = df_cat.drop_duplicates().reset_index()
    del df_cat["index"]
    df_cat.to_csv(f'{path_myfolder}df/sep_mini_anxiety.tsv.gz', sep='\t',
                         encoding='utf-8', compression='gzip', index=False)
    return df_cat


def get_o(i, df_results, df_variable, set_3c, set_3d, set_3f, mini_df, set_columns):
    print('======================================')
    # Replace none_values with np.nan in df
    none_value = ['"$4"', '"$5"', '"$6"', '"$7"', '$4', '$5', '$6', '$7']
    df_results[df_results.isin(none_value)] = np.nan
    for index, row in df_variable.iterrows():
        if 'voelde me moe' in row['definition_nl'] or 'was gauw moe' in row['definition_nl'] or 'voelde ik me uitgeput' in row['definition_nl']:
            print(row['variable_name'])
            print(row['definition_nl'])
            set_columns.add(row['variable_name'].replace(f'covt{i}', ''))
            set_3c.add(row['variable_name'])
            df_results[[row['variable_name']]] = df_results[[row['variable_name']]].apply(pd.to_numeric)
            df_results[row['variable_name']][df_results[row['variable_name']] <= 5] = 1
            df_results[row['variable_name']][df_results[row['variable_name']] > 5] = 0
            # print(df_results[['project_pseudo_id', row['variable_name']]])
            mini_df = pd.merge(mini_df, df_results[['project_pseudo_id', row['variable_name']]], on=['project_pseudo_id'], how='outer')
        if 'dag moeilijk concentreren of moeilijk beslissingen nemen' in row['definition_nl']:
            print(row['variable_name'])
            print(row['definition_nl'])
            set_columns.add(row['variable_name'].replace(f'covt{i}', ''))
            set_3d.add(row['variable_name'])
            # mini_df = pd.merge(mini_df, df_results[['project_pseudo_id', row['variable_name']]], on=['project_pseudo_id'], how='outer')
        if 'nacht slaapproblemen gehad' in row['definition_nl']:
            print(row['variable_name'])
            print(row['definition_nl'])
            set_columns.add(row['variable_name'].replace(f'covt{i}', ''))
            set_3f.add(row['variable_name'])
            # mini_df = pd.merge(mini_df, df_results[['project_pseudo_id', row['variable_name']]], on=['project_pseudo_id'], how='outer')
    return mini_df, set_3c, set_3d, set_3f, set_columns


def get_other_quest_anxiety(path_myfolder, path_variables, path_results, path_enumerations, mini_df):
    set_3c = set()
    set_3d = set()
    set_3f = set()
    set_columns = set()

    for i in range(1,29):
        if i < 10:
            i = f'0{i}'
        if i == 15 or i == 16:
            df_variable = pd.read_csv(f'{path_variables}covq_q_t{i}_variables.csv', sep=',', encoding='utf-8')
            df_results = pd.read_csv(f'{path_results}covq_q_t{i}_results.csv', sep=',', encoding='utf-8')
            df_enumerations = pd.read_csv(f'{path_enumerations}covq_q_t{i}_enumerations.csv', sep=',', encoding='utf-8')
            mini_df, set_3c, set_3d, set_3f, set_columns = get_o(i, df_results, df_variable, set_3c, set_3d, set_3f, mini_df, set_columns)
            # b
            df_variable = pd.read_csv(f'{path_variables}covq_q_t{i}b_variables.csv', sep=',', encoding='utf-8')
            df_results = pd.read_csv(f'{path_results}covq_q_t{i}b_results.csv', sep=',', encoding='utf-8')
            df_enumerations = pd.read_csv(f'{path_enumerations}covq_q_t{i}b_enumerations.csv', sep=',', encoding='utf-8')
            mini_df, set_3c, set_3d, set_3f, set_columns = get_o(f'{i}b', df_results, df_variable, set_3c, set_3d, set_3f, mini_df, set_columns)

        else:
            df_variable = pd.read_csv(f'{path_variables}covq_q_t{i}_variables.csv', sep=',', encoding='utf-8')
            df_results = pd.read_csv(f'{path_results}covq_q_t{i}_results.csv', sep=',', encoding='utf-8')
            df_enumerations = pd.read_csv(f'{path_enumerations}covq_q_t{i}_enumerations.csv', sep=',', encoding='utf-8')
            mini_df, set_3c, set_3d, set_3f, set_columns = get_o(i, df_results, df_variable, set_3c, set_3d, set_3f, mini_df, set_columns)
            
    # print(set_3f)
    print(list(mini_df.columns))
    print(set_columns)

    return mini_df, list(set_3c), list(set_3d), list(set_3f)


def get_data(MINI_path, mini_df):
    if mini_df.empty:
        mini_df = pd.read_csv(f'{MINI_path}/MINI_test.tsv.gz', sep='\t', encoding='utf-8',
                                compression='gzip')
    cat_columns = ['project_pseudo_id']
    for col in mini_df.columns:
        if 'mini' in col:
            # print(col)
            cat_columns.append(col)
    df_cat = mini_df[cat_columns]
    df_cat = df_cat.drop_duplicates().reset_index()
    del df_cat["index"]
    print(df_cat)
    print(df_cat.columns)
    # df_cat.to_csv(f'{MINI_path}sep_mini_depressive.tsv.gz', sep='\t',
    #                      encoding='utf-8', compression='gzip', index=False)
    return df_cat

def add_mini(path_directory, MINI_path):
    mini_df = pd.DataFrame(columns=['project_pseudo_id'])
    # print(new_df)
    interesting_column = '_mini'
    interesting_column_2 = '_symptoms_adu_q_1_q'

    
    for files in os.listdir(path_directory):
        if files.startswith('cov'):
            print(files)
            df = pd.read_csv(f'{path_directory}{files}', sep=',', encoding='utf-8')
            for col in df.columns:
                if 'responsedate' in col:
                    print(col)
                    response_col = col
            # Replace none_values with np.nan in df
            none_value = ['"$4"', '"$5"', '"$6"', '"$7"', '$4', '$5', '$6', '$7']
            df[df.isin(none_value)] = np.nan
            for col in df.columns:  
                if interesting_column in col or interesting_column_2 in col:
                    # print(col)
                    # print(set(df[col]))
                    # variables_df = pd.read_csv(f"{config['path_questionnaire_enumerations']}{files.replace('results', 'enumerations')}", sep=',', encoding='utf-8')
                    # select_var = variables_df[variables_df['variable_name'] == col]
                    # print(select_var)
                    # for index, row in select_var.iterrows():
                    #     print(row['enumeration_en'])
                    df[col] = df[col].astype(str).replace('nan', np.nan)
                    # print(set(df[col]))
                    if len(mini_df) < 1:
                        mini_df = df[['project_pseudo_id', col, response_col]]
                    else:
                        col_dup = list(set(df[['project_pseudo_id', col, response_col]].columns) & set(mini_df.columns))
                        mini_df = pd.merge(mini_df, df[['project_pseudo_id', col, response_col]], on=col_dup, how="outer")

    print(mini_df)
    mini_df.to_csv(f'{MINI_path}MINI_test.tsv.gz', sep='\t',
                         encoding='utf-8', compression='gzip', index=False)
    return mini_df
    
    
 
    

def main():
    config = get_config()
    my_folder = config['my_folder']
    path_myfolder = config['path_read_QOL']
    path_variables = config['path_questionnaire_variables']
    path_results = config['path_questionnaire_results']
    path_enumerations = config['path_questionnaire_enumerations']
    MINI_path = config['MINI']

    mini_df = pd.DataFrame()
    mini_df = add_mini(path_results, MINI_path)
    print('begin met functies')
    mini_df = get_data(MINI_path, mini_df)
    # print('calculating_depressive')
    # df_mini_depressive = calculating_depressive(path_myfolder, mini_df)
    # print('get_other_quest_anxiety')
    # mini_df, set_3c, set_3d, set_3f =get_other_quest_anxiety(path_myfolder, path_variables, path_results, path_enumerations, mini_df)
    # print('calculating_anxiety')
    # df_mini_anxiety = calculating_anxiety(path_myfolder, mini_df, set_3c, set_3d, set_3f)
    # print('merge')
    # mini_df_all = pd.merge(df_mini_depressive, df_mini_anxiety, on=['project_pseudo_id'], how='outer')
    # mini_df_all.to_csv(f'{path_myfolder}df/sep_mini_depressive_anxiety.tsv.gz', sep='\t',
    #                      encoding='utf-8', compression='gzip', index=False)
    # print(mini_df)
    print('DONE')


if __name__ == '__main__':
    main()

