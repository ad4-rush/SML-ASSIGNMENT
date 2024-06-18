import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Load the MNIST dataset (you may need to adjust the data loading based on your actual dataset)
# Assuming X_train and y_train are your training data and labels
# Assuming X_test and y_test are your test data and labels
# Make sure your data is properly loaded before running this code
# test_size= 10000
mnist = np.load(r'C:\Users\adars\Downloads\Adarsh_2022024\Adarsh_2022024\mnist.npz')
X, y = mnist['x_train'], mnist['y_train']


# Remove mean from X
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean
# Create PCA object
pca = PCA(n_components=700) 

# Flatten each 28x28 image to a 1D array (784 elements)
X_flattened = X_centered.reshape(X_centered.shape[0], -1)

# Apply PCA on the flattened data
pca.fit(X_flattened)

# Transform the data to the lower-dimensional space
X_pca = pca.transform(X_flattened)

# Choose p = 5, 10, 20 eigenvectors from U
for p in [1,5,10,20,700]:
    Up = pca.components_[:p, :]
    # print(Up.shape,X_pca.shape)
    Yp = np.dot(X_centered.reshape(X_centered.shape[0], -1), Up.T)  # Flatten X_centered before the dot product
    X_reconstructed_p = np.dot(Yp, Up) + X_mean.flatten()  # Flatten X_mean before addition
    print(X_reconstructed_p.shape)
    X_reconstructed_p = X_reconstructed_p.reshape(-1, 28, 28)    # Reshape each column to 28x28

    # Plot 5 images from each class
    fig, axes = plt.subplots(10, 5, figsize=(10, 15))
    for i in range(10):
        indices = np.where(y == i)[0][:5]
        for j, idx in enumerate(indices):
            axes[i, j].imshow(X_reconstructed_p[idx], cmap='gray')
            axes[i, j].axis('off')
    plt.suptitle(f'Reconstructed Images with {p} Eigenvectors')
    plt.savefig(f'reconstructed_p{p}.png')
    plt.show()

