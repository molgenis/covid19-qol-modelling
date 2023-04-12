#!/usr/bin/bash

# ---------------------------------------------------------
# Author: Anne van Ewijk
# University Medical Center Groningen / Department of Genetics
#
# Copyright (c) Anne van Ewijk, 2023
#
# ---------------------------------------------------------

ml Anaconda3/5.3.0
#source activate covid
ml Python/3.9.1-GCCcore-7.3.0-bare
ml R/4.0.3-foss-2018b-bare

cd /groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/covid19-qol-modelling/src/python/

echo "ALL BEGIN"

echo "make_df_id.py"
python3 make_df_id.py
echo "calculate_beta.py"
python3 calculate_beta.py
echo "question_15_or_more.py"
python3 question_15_or_more.py
echo "create_model.py"
python3 create_model.py
echo "information_num_quest.py"
python3 information_num_quest.py
echo "BFI.py"
python3 BFI.py
echo "resilience.py"
python3 resilience.py
echo "head_top_null_beta.py"
python3 head_top_null_beta.py
echo "mini_data.py"
python3 mini_data.py
echo "create_file_with_groups.py"
python create_file_with_groups.py
echo "tests_over_groups_and_beta.py"
python3 tests_over_groups_and_beta.py

echo "ALL DONE"
