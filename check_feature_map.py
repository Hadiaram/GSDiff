import numpy as np
import sys

file_path = 'datasets/prerunning_cnn_featuremaps/0.npy'
data = np.load(file_path, allow_pickle=True).item()

print(f"Type: {type(data)}")
if isinstance(data, dict):
    print(f"Keys: {list(data.keys())}")
    print(f"Has key 16: {16 in data}")
    for key in data.keys():
        print(f"Key {key}: type={type(data[key])}, shape={data[key][0].shape if hasattr(data[key], '__getitem__') else 'N/A'}")
else:
    print("Not a dictionary!")
