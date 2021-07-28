import pandas as pd
import numpy as np


def read_elements_from_file(elements_file):
    """
    Read elements list from file.

    :param str elements_file: file name with elements
    :return: list of elements, amount of elements
    """
    elements = open(elements_file, 'r').readlines()

    count = 1
    elements_list = []
    for element in elements:
        elements_list.append(element.strip())
        count += 1

    return elements_list, count


def describe_elements(structure_in_lines):
    """
    Find predefined elements in structure and write their parameters.

    :param list structure_in_lines: opened structure in lines
    :return: list of elements parameters, list of the last parameters
    """
    elements, count = read_elements_from_file("MADX\quads.txt")

    elements_description = []
    elements_parameters = []

    for element in elements:
        element = element.split()[0]

        for line in structure_in_lines:
            if line.startswith(element):
                line = line.replace(',', '').replace(';', '').split()
                elements_description.append(line)
                elements_parameters.append(float(line[-1]))

    elements_parameters = np.array(elements_parameters)

    return elements_description, elements_parameters


def read_BPMs(BPMs_data):
    """
    Read data from BPMs.

    :param str BPMs_data: file name with BPMs data
    :return: float X, Y beam orbits
    """
    data = pd.read_csv(BPMs_data, sep='\t')
    assert data.shape != 54

    X = data.iloc[:,1:3].to_numpy()
    Y = data.iloc[:,1:4:2].to_numpy()

    return X, Y


