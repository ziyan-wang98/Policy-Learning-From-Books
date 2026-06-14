

import numpy as np
import io


def load_inner_npz(data, inner_file_key):
    if inner_file_key in data:
        npz_binary = data[inner_file_key]
        npz_data = np.load(io.BytesIO(npz_binary), allow_pickle=True)
        return npz_data
    else:
        raise KeyError(f"{inner_file_key} not found")

def load_inner_npz_by_index(data, index):
    keys = list(data.keys())  # Convert the keys to a list
    if index < len(keys):
        inner_file_key = keys[index]  # Get the key at the specified index
        npz_binary = data[inner_file_key]
        npz_data = np.load(io.BytesIO(npz_binary), allow_pickle=True)
        return npz_data
    else:
        raise IndexError(f"Index {index} is out of bounds for the number of files.")
