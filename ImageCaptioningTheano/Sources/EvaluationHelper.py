import numpy as np

def EmpError(y, expected_y, axis=None):
    """ Calculate empirical error given y (model's output) and expected_y
    :type y: integer vector (1D) or confident matrix (4D)
    :param y: model's output
        if it is a 4D tensor, the first dimension should indicate n_sample

    :type test_output: integer vector (1D) or confident matrix (4D)
    :param test_output: what we want y to be

    """

    assert (y.shape == expected_y.shape), "Model output has different shape from expected output"

    n_sample = y.shape[0]

    # 1D (integer) label vector
    if (len(y.shape) == 1):
        return (np.sum(y != expected_y))

    # 4D tensor, could be float or integer
    elif (len(y.shape) == 4):
        if (axis == None):
            labels = np.argmax(y, axis=1)
            expected_label = np.argmax(expected_y, axis=1)
        else:
            labels = np.argmax(y, axis=axis)
            expected_label = np.argmax(expected_y, axis=axis)
        return (np.sum(labels != expected_label))

    else:
        assert(True), "Model output should only be 1D or 4D"
