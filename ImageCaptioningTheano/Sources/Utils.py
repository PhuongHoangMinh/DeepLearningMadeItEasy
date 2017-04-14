import cPickle
import sys
import h5py
import numpy as np
sys.setrecursionlimit(10000)

def SaveList(list_obj, file_name, type='pickle'):
    """ Save a list object to file_name
    :type list_obj: list
    :param list_obj: List of objects to be saved.*

    :type file_name: str
    :param file_name: file name

    :type type: str
    :param type: 'pickle' or 'hdf5'
    """

    if (type == 'pickle'):
        f = open(file_name, 'wb')
        for obj in list_obj:
            cPickle.dump(obj, f, cPickle.HIGHEST_PROTOCOL)
        f.close()
    # elif (type == 'hdf5' or type == 'h5'):
    #     with h5py.File(file_name, 'w') as hf:
    #         hf.create_dataset('data', data=list_obj)
    else:
        print('Encoding type not recognized, type should be \'pickle\' or \'hdf5\'')


def LoadList(file_name, type='pickle'):
    """ Load a list object to file_name
    :type file_name: str
    :param file_name: file name

    :type type: str
    :param type: 'pickle' or 'hdf5'
    """
    if (type == 'pickle'):
        end_of_file = False
        list_obj = []
        f = open(file_name, 'rb')
        while (not end_of_file):
            try:
                list_obj.append(cPickle.load(f))
            except EOFError:
                end_of_file = True
                print("EOF Reached")

        f.close()
        return list_obj
    # elif (type == 'hdf5' or type == 'h5'):
    #     with h5py.File(file_name, 'r') as hf:
    #         list_obj = hf.get('data')
    #         return list_obj
    else:
        print('Encoding type not recognized, type should be \'pickle\' or \'hdf5\'')


def SaveH5(obj, file_name):
    """ Save numpy data to HDF5 file
    Use this when cpickle can't save large file
    :type obj: dict
    :param obj: dict of numpy arrays

    :type file_name: str
    :param file_name: file name
    """
    with h5py.File(file_name, 'w') as hf:
        for k, v in obj.iteritems():
            hf.create_dataset(k, data=v)

def LoadH5(file_name):
    """ Load numpy data from HDF5 file
    :type obj: dict
    :param obj: dict of numpy arrays

    :type file_name: str
    :param file_name: file name
    """
    obj = {}
    with h5py.File(file_name, 'r') as hf:
        for k in hf.keys():
            obj[k] = np.asarray(hf.get(k))

    return obj
