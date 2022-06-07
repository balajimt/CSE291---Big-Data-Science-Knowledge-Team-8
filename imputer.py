import numpy as np
from tqdm import tqdm
import pandas as pd
import sys
import os
import sklearn.neighbors._base

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

from collections import defaultdict
from missingpy import MissForest

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import pickle


class Utils:
    # Constants for output locations, change per iteration
    CLUSTER_OUTPUT_FILE = 'canonical_clusters.pkl'
    IMPUTED_OUTPUT_FILE = 'main_imputed_new.csv'
    INDIVIDUAL_INPUT_FOLDER = 'individual_impute_4'
    ERROR_FILE = 'error_files.pkl'

    @staticmethod
    def intersection(lst1, lst2):
        """
        Function to get intersection between two lists

        :param lst1: List 1
        :param lst2: List 2
        :return: Intersection list between list 1 and list 2
        """
        return list(set(lst1) & set(lst2))

    @staticmethod
    def impute_values(df):
        """
        Imputer function that fits values based on a single cluster df

        :param df: Input cluster df with missing values
        :return: Output cluster df with imputed values
        """
        imputer = MissForest()
        X_fit = imputer.fit_transform(df)
        return pd.DataFrame(X_fit, df.index)

    @staticmethod
    def preprocessing(variants):
        """
        Preprocessing stub - consistent across all models
        :param variants: Input df
        :return: Pre-processed df
        """
        variants.replace(0.0, np.nan, inplace=True)
        variants_processed = variants[
            ['Peptide'] + [c for c in variants.columns if 'intensity_for_peptide_variant' in c]
            ]
        variants_processed.replace(0.0, np.nan, inplace=True)
        variants_processed = variants_processed.set_index('Peptide')
        variants_processed = variants_processed.T
        variants_processed.index = variants_processed.index.map(lambda x: '.'.join(x.split('.')[:2]))
        variants_processed['Condition'] = variants_processed.index.map(lambda x: x.split('.')[0])
        variants_processed['OG Class'] = variants_processed['Condition']
        variants_processed['OG Class'].replace(
            {"_dyn_#Severe-COVID-19": "Severe-COVID-19", "_dyn_#Non-severe-COVID-19": "Non-severe-COVID-19",
             "_dyn_#Healthy": "Healthy", "_dyn_#Symptomatic-non-COVID-19": "Symptomatic-non-COVID-19"},
            inplace=True)
        variants_processed['Condition'].replace({"_dyn_#Severe-COVID-19": "Covid", "_dyn_#Non-severe-COVID-19": "Covid",
                                                 "_dyn_#Healthy": "No-Covid",
                                                 "_dyn_#Symptomatic-non-COVID-19": "No-Covid"},
                                                inplace=True)
        variants_processed = variants_processed[(variants_processed.Condition == "Covid")
                                                | (variants_processed.Condition == "No-Covid")]
        variants_processed = variants_processed.dropna(thresh=1, axis=1)
        return variants_processed

    @staticmethod
    def combine_individual_files(folder):
        """
        Module to combine individually generated csv files
        :param folder: Input folder
        """
        INDIVIDUAL_IMPUTE = folder
        arr = os.listdir(INDIVIDUAL_IMPUTE)
        df_list = []
        df = pd.read_csv(os.path.join(INDIVIDUAL_IMPUTE, arr[0]))
        patients = df['Unnamed: 0']
        for i in arr:
            df = pd.read_csv(os.path.join(INDIVIDUAL_IMPUTE, i))
            df.drop(columns=['Unnamed: 0'], inplace=True)
            df = df.add_prefix(i + '_')
            df_list.append(df)
        df_main = pd.concat(df_list, axis=1)
        df_main['Patients'] = patients
        df_main = df_main.set_index('Patients')
        df_main['Condition'] = df_main.index.map(lambda x: x.split('.')[0])
        df_main['OG Class'] = df_main['Condition']
        df_main['OG Class'].replace(
            {"_dyn_#Severe-COVID-19": "Severe-COVID-19", "_dyn_#Non-severe-COVID-19": "Non-severe-COVID-19",
             "_dyn_#Healthy": "Healthy", "_dyn_#Symptomatic-non-COVID-19": "Symptomatic-non-COVID-19"},
            inplace=True)

        df_main['Condition'].replace({"_dyn_#Severe-COVID-19": "Covid", "_dyn_#Non-severe-COVID-19": "Covid",
                                      "_dyn_#Healthy": "No-Covid", "_dyn_#Symptomatic-non-COVID-19": "No-Covid"},
                                     inplace=True)
        df_main = df_main[(df_main.Condition == "Covid") | (df_main.Condition == "No-Covid")]
        df_main['Condition'].replace(['Covid', 'No-Covid'], [0, 1], inplace=True)
        df_main.to_csv(Utils.IMPUTED_OUTPUT_FILE)
        print("IMPUTED CSV SHAPE", df_main.shape)

    @staticmethod
    def dump_keys(canonical_list):
        """
        Util function to dump cluster names list to a file
        :param canonical_list: List of canonical clusters
        """
        with open(Utils.CLUSTER_OUTPUT_FILE, 'wb') as f:
            pickle.dump(canonical_list, f)

    @staticmethod
    def get_clusters():
        """
        Util function to retrieve cluster names from the saved file
        """
        with open(Utils.CLUSTER_OUTPUT_FILE, 'rb') as f:
            canonical_list = pickle.load(f)
        return canonical_list

    @staticmethod
    def format_csv(CSV_PATH):
        """
        Function to convert main csv file consistent with the original dataset
        The imputed csv file will contain column names cluster wise, this function
        formats it into the original column names
        :param CSV_PATH: Saved path of imputed csv file
        """
        print('STARTING TO FORMAT')
        variants = pd.read_csv(CSV_PATH)
        print(variants.shape)
        canonical_list = Utils.get_clusters()
        column_names = variants.columns.to_list()
        print('STARTING TO PROCESS COLUMNS')
        length = len(column_names)
        column_rename_dict = {}
        for column_name_index in tqdm(range(length)):
            column_name = column_names[column_name_index]
            if column_name.startswith('sp') or column_name.startswith('TRY'):
                cluster_name, index = Utils.get_cluster_and_index(column_name)
                try:
                    column_name_original = canonical_list[cluster_name][index]
                    column_rename_dict[column_name] = column_name_original
                except Exception as e:
                    print(cluster_name, index)
        print('STARTING TO RENAME COLUMN')
        variants = variants.rename(columns=column_rename_dict)
        print(variants.shape)
        new_name = CSV_PATH.replace('.csv', '') + '_renamed_2.csv'
        variants.to_csv(new_name)
        print("FORMATTING COMPLETE SAVED TO", new_name)

    @staticmethod
    def get_cluster_and_index(string_name: str):
        """
        Function to convert cluster name column to formatted cluster name and index
        Example input: sp_P05452_TETN_HUMAN.csv_3
        :param string_name: Cluster name from imputed file
        :return: original cluster name, index
        """
        items = string_name.split('.csv')
        cluster_name = items[0].replace("_", "|").replace('|HUMAN','_HUMAN').replace('TRYP|PIG','TRYP_PIG').replace('TRY1|BOVIN','TRY1_BOVIN')
        index = int(items[1].replace('_', ''))
        return cluster_name, index


def main():
    """
    Main control function
    """
    INPUT_MAESTRO_DATA = "dataset/MAESTRO-d6178bdd-identified_variants_merged_protein_regions-main.tsv"
    variants = pd.read_csv(INPUT_MAESTRO_DATA, sep="\t", low_memory=False)
    variants_processed = Utils.preprocessing(variants)

    canonical_list = defaultdict(list)
    new = variants[['Peptide', 'Canonical_proteins']]
    for index, row in new.iterrows():
        canonical_list[row['Canonical_proteins']].append(row['Peptide'])

    keys = list(canonical_list.keys())
    # # todo: FOLDER TO GENERATE INDIVIDUAL CSVs --update if changes are required
    Path(Utils.INDIVIDUAL_INPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    error_files = []

    arr = os.listdir(Utils.INDIVIDUAL_INPUT_FOLDER)

    list2 = list(variants_processed.columns)
    for key in tqdm(keys):
        if type(key) != str:
            continue
        new_name = key.replace("|", "_")
        if new_name in arr:
            continue
        try:
            list1 = canonical_list[key]
            individual_csv = Utils.impute_values(variants_processed[Utils.intersection(list1, list2)])
            individual_csv.to_csv(os.path.join(Utils.INDIVIDUAL_INPUT_FOLDER, new_name + ".csv"))
        except Exception as e:
            print(e)
            error_files.append(key)
            continue

    with open(Utils.ERROR_FILE, 'wb') as f:
        pickle.dump(error_files, f)

    with open(Utils.ERROR_FILE, 'rb') as f:
        new_error_list = pickle.load(f)

    print("ERROR files", len(new_error_list))

    Utils.combine_individual_files(Utils.INDIVIDUAL_INPUT_FOLDER)

    Utils.dump_keys(canonical_list)
    Utils.format_csv(Utils.IMPUTED_OUTPUT_FILE)


main()
