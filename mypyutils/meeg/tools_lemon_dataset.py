import os
import pandas
import numpy as np


def select_subjects(age_gr, gender_gr, handedness_gr, meta_file_path):
    # meta_file_path = '/data/pt_02076/LEMON/INFO/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'
    # meta = pandas.read_csv(meta_file_path, sep=',')
    age = id_meta(meta_file_path, value='Age')
    id = id_meta(meta_file_path, value='ID')
    gender = id_meta(meta_file_path, value='Gender')
    handedness = id_meta(meta_file_path, value='Handedness')
    if age_gr == 'young':
        ind_age = (age == '20-25') + (age == '25-30') + (age == '30-35')
        ind_age = ind_age > 0
    if gender_gr == 'male':
        ind_gender = gender == 2
    ind_handedness = handedness == handedness_gr

    ind_final = (ind_handedness+0) + (ind_age+0) + (ind_gender+0)
    ind_final = ind_final == 3

    return id[ind_final]


def id_meta(meta_file_path, value=None):
    # meta_file_path = '/data/pt_02076/LEMON/INFO/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'
    meta = pandas.read_csv(meta_file_path, sep=',')
    if value is None:
        return meta
    else:
        value_col = [i for i in range(10) if value in meta.columns[i]]
        if len(value_col):
            return meta.values[:, value_col]
        else:
            print('value is not valid!')

# def id_specify_age()


def select_subject_1():
    """
    select subjects with standard electrode positions from a specific file path_elec
    :return:
    """
    id = id_meta('ID')
    gender = id_meta('Gender')
    age = id_meta('Age')

    path_elec = '/data/pt_02076/LEMON/INFO/electrode_status_reorder_Mina.csv'
    elec_stat = pandas.read_csv(path_elec, sep=',')
    id_elec = elec_stat.values[:, 0]
    stat_elec = elec_stat.values[:, 1]

    id_elec_in_id = np.zeros((len(id_elec), 1))
    for k in range(len(stat_elec)):
        id_elec_in_id[k] = [i for i in range(len(id)) if str(id_elec[k]) in id[i][0]]

    id_use = id[id_elec_in_id.astype('int'), 0]
    gender_use = gender[id_elec_in_id.astype('int'), 0]
    age_use = age[id_elec_in_id.astype('int'), 0]
    ind_young = [i for i in range(len(age_use)) if
                 '20-25' in age_use[i] or '25-30' in age_use[i] or '30-35' in age_use[i]]
    return id_use, gender_use, age_use


def subjects_in_directory(path):
    import fnmatch
    all_ids = os.listdir(path)
    all_ids = np.sort(all_ids)
    all_ids = list(all_ids[k][:10] for k in range(len(all_ids))
                            if fnmatch.fnmatch(all_ids[k], 'sub-*'))
    all_ids = np.unique(np.asarray(all_ids))
    return all_ids
