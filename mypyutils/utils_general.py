
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import matplotlib.colors as colors

#  --------------------------------  --------------------------------  --------------------------------
# general: strings, saving, loading, directories
#  --------------------------------  --------------------------------  --------------------------------


def strround(x, n=3):
    return str(np.round(x, n))


def combine_names(connector, *nouns):
    word = nouns[0]
    for noun in nouns[1:]:
        word += connector + str(noun)
    return word


def save_pickle(file, var):
    import pickle
    with open(file, "wb") as output_file:
        pickle.dump(var, output_file)


def load_pickle(file):
    import pickle
    with open(file, "rb") as input_file:
        var = pickle.load(input_file)
    return var


def np_parsave(save_name, var):
    np.save(save_name, var)


def write_in_txt(filename, msg):
    """
    the function to write a message in a txt file
    Input arguments:
    ================
    filename: the name of the file, e.g 'file1.txt'
    msg: the message to be writte

    Output arguments:
    =================
    No value returns. The file is modified
    """
    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    lines.append(msg+'\n')
    f = open(filename, "w")
    for line in lines:
        f.write(line)
    f.close()


def listdir_restricted(dir_path, string_criterion, startswith=None, endswith=None, path=False):
    """
    returns a list of the names of the files in dir_path, whose names contains string_criterion
    :param dir_path: the directory path
    :param string1: a list of strings that we wanna be included in the name of the files

    :return: list of file names
    """
    import os
    IDs_all = os.listdir(dir_path)

    def id_name(path, id, dir_path):
        if path:
            return op.join(dir_path, id)
        else:
            return id

    if startswith:
        IDs_with_string = [id_name(path, id1, dir_path) for id1 in IDs_all if id1.startswith(string_criterion)]
    elif endswith:
        IDs_with_string = [id_name(path, id1, dir_path) for id1 in IDs_all if id1.endswith(string_criterion)]
    else:
        IDs_with_string = [id_name(path, id1, dir_path) for id1 in IDs_all if string_criterion in id1]

    return IDs_with_string


def save_json_from_numpy(filename, var):
    """
    (c) by Alina Studenova - 2021
    save numpy array as json file
    :param filename:
    :param var:
    :return:
    """
    import json
    with open(filename, "w") as f:
        json.dump(var.tolist(), f)


def load_json_to_numpy(filename):
    """
    (c) by Alina Studenova - 2021

    load json file to a numpy array
    :param filename:
    :return:
    """
    import json
    with open(filename, "r") as f:
        saved_data = json.load(f)

    var = np.array(saved_data)
    return var


def save_json(filename, var):
    """
    (c) by Alina Studenova - 2021

    save list to json
    :param filename:
    :param var:
    :return:
    """
    import json
    with open(filename, "w") as f:
        json.dump(var, f)


def load_json(filename):
    """
    (c) by Alina Studenova - 2021

    load list from json
    :param filename:
    :return:
    """
    import json
    with open(filename, "r") as f:
        saved_data = json.load(f)

    var = saved_data
    return var

def combine_names(connector, *nouns):
    word = nouns[0]
    for noun in nouns[1:]:
        word += connector + str(noun)
    return word

