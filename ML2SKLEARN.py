import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

doc = pd.read_csv(r'D:/dữ liệu học máy, R/mall_chuan_hoa.csv')
print(doc)


# Lấy cột 'Annual Income (k$)' và 'Spending Score (1-100)'
X = doc[['Thu nhập_Normalized', 'Chi tiêu_Normalized']]

# Số lượng cụm
k = 3

# Sử dụng KMeans từ thư viện sklearn
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Nhãn của các cụm
labels = kmeans.labels_

# Vị trí của các centroids
centers = kmeans.cluster_centers_

#tính sai số cho mô hình
sed = 0
for i in range(len(X)):
    cluster_center = centers[labels[i]]
    sed += np.linalg.norm(X.iloc[i] - cluster_center)**2

print("Tổng Squared Euclidean Distance:", sed)


# Vẽ biểu đồ phân cụm
colors = ['r', 'g', 'b']
for i in range(k):
    plt.scatter(X.iloc[labels == i, 0], X.iloc[labels == i, 1], c=colors[i], label=f'Cluster {i+1}')
plt.scatter(centers[:, 0], centers[:, 1], marker='.', s=300, c='black', label='centers')
plt.xlabel('Mức thu nhập)')
plt.ylabel('Mức chi tiêu')
plt.title('Phân cụm khách hàng')
plt.legend()
plt.show()