from tarfile import open as taropen
from struct import unpack
from pandas import DataFrame
from numpy.random import permutation
from math import sqrt

MODKEY = "mod"
RMS_MAX_DIFF_LABEL = "rms_max_diff"
RMS_MAX_RATIO_LABEL = "rms_max_ratio"

def data_append_floats(data, floats):
    # calculate and add cross features
    aplitude_max = abs(max(floats))
    aplitude_rms = sqrt(sum([f*f for f in floats])/len(floats))
    if not RMS_MAX_DIFF_LABEL in data:
        data[RMS_MAX_DIFF_LABEL] = [aplitude_max-aplitude_rms]
    else:
        data[RMS_MAX_DIFF_LABEL].append(aplitude_max-aplitude_rms)
    if not RMS_MAX_RATIO_LABEL in data:
        data[RMS_MAX_RATIO_LABEL] = [aplitude_max/aplitude_rms] if 0 != aplitude_rms else [1]
    else:
        data[RMS_MAX_RATIO_LABEL].append(aplitude_max/aplitude_rms if 0 != aplitude_rms else 1)

    # seperate imaginary and quadrature parts
    i = floats[0::2][::4]
    q = floats[1::2][::4]
    for j in range(min(len(i), len(q))):
        ikey = "i{:05d}".format(j)
        qkey = "q{:05d}".format(j)
        if not ikey in data:
            data[ikey] = [i[j]]
        else:
            data[ikey].append(i[j])
        if not qkey in data:
            data[qkey] = [q[j]]
        else:
            data[qkey].append(q[j])

    return data

def load_data(files):
    num_classes = len(files)
    data = dict()
    for modulation in range(num_classes):
        tar = taropen(files[modulation])
        for member in tar.getmembers():
            if not MODKEY in data:
                data[MODKEY] = [modulation]
            else:
                data[MODKEY].append(modulation)
            with tar.extractfile(member) as f:
                # load float pairs from file
                buffer = f.read()
                num_floats = len(buffer)//4
                floats = unpack("f"*num_floats, buffer)
                data = data_append_floats(data, floats)
    signal_dataframe = DataFrame(data=data)
    signal_dataframeReal = signal_dataframe.copy()
    signal_dataframe = signal_dataframe.reindex(permutation(signal_dataframe.index))
    return signal_dataframe
