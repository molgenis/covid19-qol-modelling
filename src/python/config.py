#!/usr/bin/env python3

# ---------------------------------------------------------
# Author: Anne van Ewijk
# University Medical Center Groningen / Department of Genetics
#
# Copyright (c) Anne van Ewijk, 2023
#
# ---------------------------------------------------------

# Imports
import yaml


def get_config():
    """
    Get the configuration of personalized variable definitions
    # :param cluster_name: Name of the cluster
    :return: config:     Dictionary with as keys the name of the paths and as value the paths
    """
    with open(
            '/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/covid19-qol-modelling/src/python/config.yaml',
            'r') as stream:
        config = yaml.safe_load(stream)
    return config
