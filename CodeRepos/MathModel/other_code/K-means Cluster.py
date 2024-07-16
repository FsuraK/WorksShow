"""
这段代码适用于需要对数据进行聚类分析的场景，例如市场细分、社交网络分析、城市规划、客户细分等。
在这些场景中，我们可以通过聚类分析将大量数据划分为几个不同的组或者类别。
每个组内的数据在某种程度上是相似的，而不同组之间的数据则存在明显的差异。这样可以帮助我们更好地理解和解释数据。
例如，在市场细分中，我们可以通过聚类分析将客户划分为几个不同的群体，然后针对每个群体制定不同的营销策略。
在社交网络分析中，我们可以通过聚类分析找出社交网络中的社区结构等。在城市规划中，我们可以通过聚类分析对城市中的各种设施进行分类和布局等。
在客户细分中，我们可以通过聚类分析将客户划分为几个不同的群体，然后针对每个群体制定不同的服务策略等。所以这段代码在很多场景中都有可能被使用到。
"""

from sklearn.cluster import KMeans
import numpy as np


class ClusterAnalysis:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters)

    def fit(self, data):
        self.model.fit(data)

    def predict(self, data):
        return self.model.predict(data)

    def fit_predict(self, data):
        self.fit(data)
        return self.predict(data)


# 使用示例：
# 创建一个实例
cluster_analysis = ClusterAnalysis(n_clusters=3)

# 假设我们有一些数据
data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

# 拟合数据
cluster_analysis.fit(data)

# 预测数据
predictions = cluster_analysis.predict(data)
print(predictions)  # 输出：[1 1 1 0 0 2]
