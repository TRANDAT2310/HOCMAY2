import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


doc = pd.read_csv(r'D:/dữ liệu học máy, R/mall_chuan_hoa.csv')
# kiểm tra giá trị lỗi
doc.isnull().sum()


X = doc[['Thu nhập_Normalized','Chi tiêu_Normalized' ]].values

#số lượng cụm
k = 3

#khởi tạo ngẫu nhiên tâm ban đầu
centers = X[np.random.choice(X.shape[0],k , replace=False)]

#tính euclide giữa điểm dữ liệu và center
def  khoangcach_euclide(X,centers):
    khoangcach=[]
    for center in centers:
        khoangcach.append(np.linalg.norm(X - center, axis=1))
    return np.array(khoangcach)

#gán nhãn cho điểm dữ liệu dựa trên center gần nhất
def gan_nhan(khoangcach):
    return np.argmin(khoangcach, axis=0)

#update vị trí các center
def update_centers(X, labels, k):
    centers = []
    for i in range(k):
        centers.append(np.mean(X[labels == i], axis=0))
    return np.array(centers)

#hàm kiểm tra điều kiện dừng của thuật toán
def dieu_kien_dung(old_centers, centers, tol=1e-4):
    return np.linalg.norm(centers - old_centers) < tol

#chạy thuât toán
def k_means(X,k):
    centers = X[np.random.choice(X.shape[0], k , replace=False)]
    while True:
        old_centers =centers
        khoangcach = khoangcach_euclide(X, centers)
        labels = gan_nhan(khoangcach)
        centers = update_centers(X, labels, k)
        if dieu_kien_dung(old_centers, centers):
            break
        return labels, centers
    
def tinh_MSE(X,lables, centers):
    MSE = 0
    for i in range(k):
        cluster_point = X[labels == i]
        MSE += np.sum((cluster_point - centers[i]) **2)
    MSE /= X.shape[0]
    return MSE
#chạy 
labels, centers = k_means(X,k)

#tinh MSE
MSE = tinh_MSE(X, labels, centers)
print(f"Mean square error : {MSE}")

#vẽ biểu đồ
colors = ['r','g','b']
for i in range(k):
    plt.scatter(X[labels == i][:, 0], X[labels == i][:, 1], c=colors[i], label=f'Cluster {i+1}')
plt.scatter(centers[:,0], centers[:,1], marker='.', s=300, c= 'black', label = 'centers')
plt.xlabel('thu nhap')
plt.ylabel('chi tieu')
plt.title('phan cum khach hang')
plt.legend()
plt.show()
