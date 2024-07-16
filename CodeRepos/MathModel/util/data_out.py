import numpy as np
import os
import pandas as pd


root_dir = "/home/liuyangyang/mathModelDataSet/NJU_CPOL_update2308"
dir_name = "data_dir_033"

for variable in ["dBZ", "ZDR", "KDP"]:
    # for height in ["1.0km", "3.0km", "7.0km"]:
    for height in ["3.0km"]:
        data_path = os.path.join(root_dir, variable, height, dir_name)
        frames = sorted(os.listdir(data_path))

        for frame in frames[:10]:
            frame_path = os.path.join(data_path, frame)
            data = np.load(frame_path)
            df = pd.DataFrame(data)
            df.to_excel(frame_path+".xlsx", index=False, header=False)
