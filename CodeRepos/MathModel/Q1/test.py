import numpy as np
import os, numpy
import pandas as pd
import torch.nn as nn
# radar_data = np.load('/home/liuyangyang/mathModelDataSet/NJU_CPOL_kdpRain/data_dir_000/frame_001.npy')

data_path = '/home/liuyangyang/mathModelDataSet/NJU_CPOL_update2308/KDP/1.0km/data_dir_057'
frames = sorted(os.listdir(data_path))

for frame in frames:
    frame_path = os.path.join(data_path, frame)
    data = np.load(frame_path)
    df = pd.DataFrame(data)
    df.to_excel(r'/home/liuyangyang/mathModelDataSet/data_output/data_output/data_dir_057/' + frame[0: -4] + ".xlsx"
                , index=False, header=False)

