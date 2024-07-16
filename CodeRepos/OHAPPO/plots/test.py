import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
# 小船图像的路径
img_path = '/home/lyy/Desktop/MpeTraj_good/分散/9/boat.png'

# 加载小船图像
img = mpimg.imread(img_path)
img2 = np.rot90(img, k=1)

# 定义四个点的位置
x = [1, 2, 3, 4]
y = [3, 2, 4, 1]

# 创建画布和子图
fig, ax = plt.subplots()

# 循环将小船图像代替点
for i in range(4):
    img_obj = OffsetImage(img2, zoom=0.03)
    # 将小船图像的偏移对象添加到画布上
    ab = AnnotationBbox(img_obj, (x[i], y[i]), frameon=False, pad=0.0, alpha=1.0)
    ax.add_artist(ab)


# 设置小船图像的旋转角度


ax.set_xlim([-5, 10])
ax.set_ylim([-5, 10])
# 显示画布
plt.savefig('/home/lyy/Desktop/MpeTraj_good/分散/9/boat.pdf')
plt.show()