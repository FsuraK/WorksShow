import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

fig, ax = plt.subplots()

# 创建一个箭头标记
arrow_path = Path([(0, 0), (1, 0), (1, 0.5), (2, 0.5), (2, -0.5), (1, -0.5), (1, 0), (0, 0)], closed=True)
arrow_patch = PathPatch(arrow_path, facecolor='black', edgecolor='black')

# 创建一个散点图，并将标记样式设置为箭头
x = np.random.rand(10)
y = np.random.rand(10)
sc = ax.scatter(x, y, s=200, marker=arrow_patch, facecolor='blue', edgecolor='black', linewidth=2)

# 将标记旋转45度
arrow_patch._patch_transform.rotate_deg(45)

plt.show()