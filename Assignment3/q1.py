import numpy as np

# Load MNIST dataset
def load_mnist(url):
    with np.load(url) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return x_train, y_train, x_test, y_test

# Extract data and labels for classes 0, 1, and 2
def extract_classes(data, labels, classes):
    mask = np.isin(labels, classes)
    return data[mask], labels[mask]

# Reshape images
def reshape_images(images):
    return images.reshape((images.shape[0], -1))

# PCA implementation
def pca(X, n_components=10):
    # Centering the data
    X_centered = X - np.mean(X, axis=0)
    # Computing covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # Sorting eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    # Selecting top components
    selected_eigenvectors = sorted_eigenvectors[:, :n_components]
    # Projecting data onto selected components
    X_pca = np.dot(X_centered, selected_eigenvectors)
    return X_pca

# Decision tree implementation
class DecisionTree:
    def __init__(self, max_leaf_nodes=3):
        self.max_leaf_nodes = max_leaf_nodes
        self.tree = {}

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _compute_gini(self, y):
        classes = np.unique(y)
        n_samples = len(y)
        gini = 1.0
        for c in classes:
            p = np.sum(y == c) / n_samples
            gini -= p ** 2
        return gini

    def _split(self, X, y, split_feature, split_value):
        left_mask = X[:, split_feature] < split_value
        right_mask = ~left_mask
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def _find_best_split(self, X, y):
        best_gini = float('inf')
        best_split_feature = None
        best_split_value = None
        n_samples, n_features = X.shape
        for feature in range(n_features):
            values = np.unique(X[:, feature])
            for value in values:
                X_left, y_left, X_right, y_right = self._split(X, y, feature, value)
                gini_left = self._compute_gini(y_left)
                gini_right = self._compute_gini(y_right)
                gini = (len(y_left) / n_samples) * gini_left + (len(y_right) / n_samples) * gini_right
                if gini < best_gini:
                    best_gini = gini
                    best_split_feature = feature
                    best_split_value = value
        return best_split_feature, best_split_value

    def _grow_tree(self, X, y):
        if len(np.unique(y)) == 1 or len(y) <= self.max_leaf_nodes:
            return {'class': np.argmax(np.bincount(y))}
        split_feature, split_value = self._find_best_split(X, y)
        X_left, y_left, X_right, y_right = self._split(X, y, split_feature, split_value)
        return {'split_feature': split_feature,
                'split_value': split_value,
                'left': self._grow_tree(X_left, y_left),
                'right': self._grow_tree(X_right, y_right)}

    def predict(self, X):
        predictions = []
        for sample in X:
            node = self.tree
            while 'split_feature' in node:
                if sample[node['split_feature']] < node['split_value']:
                    node = node['left']
                else:
                    node = node['right']
            predictions.append(node['class'])
        return np.array(predictions)

# Manual accuracy computation
def compute_accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

# Bagging implementation
def bagging(train_data, train_labels, test_data, test_labels, num_datasets=5, num_trees_per_dataset=3):
    bagged_trees = []
    for _ in range(num_datasets):
        indices = np.random.choice(len(train_data), size=len(train_data), replace=True)
        train_data_bagged = train_data[indices]
        train_labels_bagged = train_labels[indices]

        trees = []
        for _ in range(num_trees_per_dataset):
            tree = DecisionTree()
            tree.fit(train_data_bagged, train_labels_bagged)
            trees.append(tree)

        bagged_trees.append(trees)

    # Classify test samples using majority voting
    def classify_with_voting(sample, trees):
        predictions = [tree.predict([sample])[0] for tree in trees]
        counts = np.bincount(predictions)
        return np.argmax(counts)

    predicted_labels_bagged = [classify_with_voting(sample, trees) for sample in test_data]
    return compute_accuracy(test_labels, predicted_labels_bagged)

# Main function
def main():
    # Load MNIST dataset
    url = "C:\Shared_archcraft\SML\Assignment3\mnist.npz"
    x_train, y_train, x_test, y_test = load_mnist(url)

    # Extract data and labels for classes 0, 1, and 2
    train_images_012, train_labels_012 = extract_classes(x_train, y_train, [0, 1, 2])
    test_images_012, test_labels_012 = extract_classes(x_test, y_test, [0, 1, 2])

    # Reshape images
    train_images_012 = reshape_images(train_images_012)
    test_images_012 = reshape_images(test_images_012)

    # Apply PCA and reduce dimension to p=10
    train_images_pca = pca(train_images_012, n_components=10)
    test_images_pca = pca(test_images_012, n_components=10)

    # Learn a decision tree
    decision_tree = DecisionTree()
    decision_tree.fit(train_images_pca, train_labels_012)

    # Classify test samples
    predicted_labels = decision_tree.predict(test_images_pca)
    decision_tree_accuracy = compute_accuracy(test_labels_012, predicted_labels)
    print("Decision Tree Accuracy:", decision_tree_accuracy)

    # Bagging
    bagging_accuracy = bagging(train_images_pca, train_labels_012, test_images_pca, test_labels_012)
    print("Bagging Accuracy:", bagging_accuracy)

if __name__ == "__main__":
    main()
