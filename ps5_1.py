import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import datasets

olivetti_faces = datasets.fetch_olivetti_faces()

fig = plt.figure(figsize=(8,8))
for i in range(64):
    ax=fig.add_subplot(8,8,i+1)
    ax.imshow(olivetti_faces.images[i],cmap=plt.cm.bone)
plt.show()
    
x = olivetti_faces.data
y = olivetti_faces.target

pca = PCA(n_components=200,whiten=True)
pca.fit(x)

transformed_data = pca.fit_transform(x)

x_approx=pca.inverse_transform(transformed_data)

x_approx_img = x_approx.reshape(400,64,64)
fig = plt.figure(figsize=(8,8))
for i in range(64):
    ax=fig.add_subplot(8,8,i+1)
    ax.imshow(x_approx_img[i],cmap=plt.cm.bone)
plt.show()

result=pd.DataFrame(pca.transform(x), columns=['PCA%i' % i for i in range(3)])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'], color='black', cmap="Set2_r", s=60)
 

xAxisLine = ((min(result['PCA0']), max(result['PCA0'])), (0, 0), (0,0))
ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
yAxisLine = ((0, 0), (min(result['PCA1']), max(result['PCA1'])), (0,0))
ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'g')
zAxisLine = ((0, 0), (0,0), (min(result['PCA2']), max(result['PCA2'])))
ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'b')
 

ax.set_xlabel("pca1")
ax.set_ylabel("pca2")
ax.set_zlabel("pca3")
ax.set_title("PCA on the Olivetti dataset ")



    

