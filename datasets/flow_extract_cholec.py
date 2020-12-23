import os
import pandas as pd
import sys
import pickle
import numpy as np
import json

# 将
def flow_extract(input_path):
    """flow_extract extract original flow and compress flow form labels

    Args:
        input_path (string): the path of current label file

    Returns:
        Array: the flow path of the labels
    """
    annotations = pd.read_csv(input_path, sep='\t')
    flow_stack = [annotations.iloc[0, :].values[1]]
    # cover csv data to [1,:] size
    label_data = np.array(annotations.iloc[:, 1:])
    print(label_data)
    full_flow = []
    for i in range(label_data.shape[0]):
        data = annotations.iloc[i, :].values[1]
        if data=='Preparation':
            data=0
            full_flow.append(data)
        elif data=='CalotTriangleDissection':
            data=1
            full_flow.append(data)
        elif data == 'ClippingCutting':
            data=2
            full_flow.append(data)
        elif data == 'GallbladderDissection':
            data=3
            full_flow.append(data)
        elif data == 'GallbladderPackaging':
            data=4
            full_flow.append(data)
        elif data == 'CleaningCoagulation':
            data=5
            full_flow.append(data)
        elif data == 'GallbladderRetraction':
            data=6
            full_flow.append(data)

        if data != flow_stack[-1]:
            flow_stack.append(data)
    return full_flow, flow_stack[1:]


def generate_tuple(compress_flow):
    """generate the flow tuple

        Args:
            compress_flow (tuple): array to extract
        Returns:
            n_tuple from data
    """
    n_tuple = []
    compress_flow_len = len(compress_flow)
    for pos in range(compress_flow_len):
        if pos == 0:
            n_tuple.append([-1, compress_flow[0], compress_flow[1]]) # 8 for not start
        elif pos == len(compress_flow) - 1:
            n_tuple.append([compress_flow[compress_flow_len - 2], compress_flow[compress_flow_len - 1], 7]) #7 for end
        else:
            n_tuple.append([compress_flow[pos - 1], compress_flow[pos], compress_flow[pos + 1]])
    return n_tuple


if __name__ == "__main__":
    abs_folder = '/home/amax/lab_data/surgicalData/cholec80_data/phase_annotations/'
    flow_data_path = os.path.join(abs_folder, "data.pkl")
    flow_data = {}
    try:
        with open(flow_data_path, "rb") as file:
            flow_data = pickle.load(file)
    except FileNotFoundError:
        with open(flow_data_path, "wb") as file:
            for labels_name in os.listdir(abs_folder):
                # if file is csv file
                if os.path.splitext(labels_name)[1] == '.txt':
                    original_flow, compress_flow = flow_extract(os.path.join(abs_folder, labels_name))
                    item = os.path.splitext(labels_name)[0].split('-')[0]
                    n_tuple = generate_tuple(compress_flow)
                    print("{} flow is {} \n"
                          "compress flow as {} \n"
                          "len {} \n"
                          "n_tuple {}".format(item, original_flow, compress_flow,
                                              len(original_flow), n_tuple))
                    sample = {
                        "frames": len(original_flow),
                        "original_flow": original_flow,
                        "compress_flow": compress_flow,
                        "n_tuple": n_tuple
                    }
                    flow_data[str(item)] = sample

            pickle.dump(flow_data, file)
    # Only for test
    # plot the phase
    # plot_3d(np.asarray(tuple(triad_set)))
