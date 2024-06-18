import numpy as np
from sklearn.decomposition import PCA

# Define a function to calculate Gini index
def calculate_gini(labels):
    n_labels = len(labels)
    if n_labels == 0:
        return 0
    counts = np.bincount(labels)
    probabilities = counts / n_labels
    gini = 1 - np.sum(probabilities ** 2)
    return gini

# Define a function to find the best split for a dimension
def find_best_split(feature, labels):
    sorted_indices = np.argsort(feature)
    best_gini = float('inf')
    best_split_value = None
    for i in range(len(feature) - 1):
        split_value = 0.5 * (feature[sorted_indices[i]] + feature[sorted_indices[i + 1]])
        # print(split_value)
        left_labels = labels[sorted_indices[:i + 1]]
        right_labels = labels[sorted_indices[i + 1:]]
        gini = (len(left_labels) * calculate_gini(left_labels) + len(right_labels) * calculate_gini(right_labels)) / len(labels)
        if gini < best_gini:
            best_gini = gini
            best_split_value = split_value
    return best_gini, best_split_value

# Define a function to grow a decision tree with 2 terminal nodes
def grow_decision_tree(features, labels, max_nodes):
    n_features = features.shape[1]
    best_split_dim = None
    best_split_value = None
    best_gini= float('inf')
    
    for dim in range(n_features):
        print(dim)
        gini, split_value = find_best_split(features[:,dim], labels)
        # print(features.shape)
        if gini < best_gini:
            best_gini = gini
            best_split_dim = dim
            best_split_value = split_value
    
    # Split the data using the best split
    left_indices = features[:, best_split_dim] <= best_split_value
    right_indices = ~left_indices
    left_labels = labels[left_indices]
    right_labels = labels[right_indices]
    left_features = features[left_indices]
    right_features = features[right_indices]
    # Compute the Gini index for each side
    gini_left = calculate_gini(left_labels)
    gini_right = calculate_gini(right_labels)
    
    if max_nodes == 0:
        node = {
            'split_dim': best_split_dim,
            'split_value': best_split_value,
            'left': {'class': np.argmax(np.bincount(left_labels))},
            'gini_left': gini_left,
            'gini_right': gini_right,
            'right': {'class': np.argmax(np.bincount(right_labels))}
        }
        return node
        
    # Choose the side with the higher Gini index and make the cut there
    if gini_left >= gini_right:
        node = {
            'split_dim': best_split_dim,
            'split_value': best_split_value,
            'left': grow_decision_tree(left_features, left_labels, max_nodes-1),
            'gini_left': gini_left,
            'gini_right': gini_right,
            'right': {'class': np.argmax(np.bincount(right_labels))}
        }
        return node
    else:
        node = {
            'split_dim': best_split_dim,
            'split_value': best_split_value,
            'left': {'class': np.argmax(np.bincount(left_labels))},
            'gini_left': gini_left,
            'gini_right': gini_right,
            'right': grow_decision_tree(right_features, right_labels, max_nodes-1)
        }
        return node
def classify_sample(sample, node):
    if 'class' in node:
        return node['class']
    elif sample[node['split_dim']] <= node['split_value']:
        return classify_sample(sample, node['left'])
    else:
        return classify_sample(sample, node['right'])


def classify_samples(samples, node):
    predictions = []
    print(samples.shape)
    for sample in samples:
        class_prediction = classify_sample(sample, node)
        predictions.append(class_prediction)
    return predictions

# Load MNIST dataset containing only classes 0, 1, and 2
with np.load("C:\Shared_archcraft\SML\Assignment3\mnist.npz") as data:
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']


def pca(x):
    
    # Remove mean from X
    X_mean = np.mean(x, axis=0)
    X_centered = x - X_mean
    # Create PCA object
    pca = PCA(n_components=10)

    # Flatten each 28x28 image to a 1D array (784 elements)
    X_flattened = X_centered.reshape(X_centered.shape[0], -1)

    # Apply PCA on the flattened data
    pca.fit(X_flattened)

    # Up = pca.components_[:10, :]
    X_pca = pca.transform(X_flattened)
    print(X_pca.shape)
    return X_pca



# Select only the samples with labels 0, 1, and 2
train_mask = np.isin(y_train, [0, 1, 2])
x_train_012 = x_train[train_mask]
y_train_012 = y_train[train_mask]

# Flatten the images
x_train_012 = x_train_012.reshape(x_train_012.shape[0], -1)

# Compute the mean of the dataset
mean_vec = np.mean(x_train_012, axis=0)

# Compute the covariance matrix
cov_mat = np.cov(x_train_012.T)

# Compute the eigenvalues and eigenvectors of the covariance matrix
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# Sort the eigenvalues in descending order
sorted_indices = np.argsort(eig_vals)[::-1]

# Select the top eigenvectors
top_eig_vecs = eig_vecs[:, sorted_indices[:10]]

# Transform the data into the reduced dimension space
x_train_reduced = pca(x_train_012)
print(x_train_reduced.shape)

def q1():
# Grow the decision tree
    node = grow_decision_tree(x_train_reduced, y_train_012, 1)
    print(node)

# Transform the test data into the reduced dimension space
def q2_helper(x_test_012, y_test_012):

    node = grow_decision_tree(x_train_reduced, y_train_012, 1)
    # mean_vec_t = np.mean(x_test_012, axis=0)
    x_test_reduced = pca(x_test_012)
    # print(x_test_reduced)

    # Classify test samples
    test_predictions = classify_samples(x_test_reduced, node)

    # Calculate accuracy
    # accuracy = np.mean(test_predictions == y_test[np.isin(y_test, [0, 1, 2])])

    # Calculate class-wise accuracy
    # Calculate total accuracy
    total_accuracy = np.mean(test_predictions == y_test_012)

    # Calculate class-wise accuracy
    class_wise_accuracy = {}
    for i in [0, 1, 2]:
        mask = y_test_012 == i
        class_accuracy = np.mean(np.array(test_predictions)[mask] == i)
        class_wise_accuracy[i] = class_accuracy

    print("Total Accuracy:", total_accuracy)
    print("Class-wise Accuracy:", class_wise_accuracy)


def q2():
    train_mask = np.isin(y_test, [0, 1, 2])
    x_test_012 = x_test[train_mask]
    y_test_012 = y_test[train_mask]
    q2_helper(x_test_012, y_test_012)
def q3():
    train_mask = np.isin(y_test, [0, 1, 2])
    x_test_012 = x_test[train_mask]

    # Define a function for majority voting
    def majority_vote(predictions):
        return np.bincount(predictions).argmax()

    # Perform bagging and train decision trees
    num_trees = 5
    trees = []
    def bootstrap_sample(x_train, y_train, sample_size=5000):
        # Generate random indices with replacement
        indices = np.random.choice(len(x_train), size=sample_size, replace=True)
        # Select samples and corresponding labels
        x_sampled = x_train[indices]
        y_sampled = y_train[indices]
        print(indices, indices.shape)
        return x_sampled, y_sampled
    
    # Define a function to make predictions using majority voting
    def predict_majority_vote(x_test, trees):
        predictions = []
        for sample in x_test:

            tree_predictions = [classify_sample(sample,tree) for tree in trees]
            majority_prediction = majority_vote(tree_predictions)
            predictions.append(majority_prediction)
        return predictions
    
    for i in range(num_trees):
        print("HI")
        x_sampled, y_sampled = bootstrap_sample(x_train_012, y_train_012)
        x_sampled = pca(x_sampled)
        tree = grow_decision_tree(x_sampled, y_sampled , 1)
        trees.append(tree)
        print(trees,"\n")


    # Classify test samples using majority voting
    test_predictions = predict_majority_vote(x_test_012, trees)

    # Calculate total accuracy
    total_accuracy = np.mean(test_predictions == y_test[np.isin(y_test, [0, 1, 2])])

    # Calculate class-wise accuracy
    class_wise_accuracy = {}
    for i in [0, 1, 2]:
        mask = y_test[np.isin(y_test, [0, 1, 2])] == i
        class_accuracy = np.mean(np.array(test_predictions)[mask] == i)
        class_wise_accuracy[i] = class_accuracy

    print("Total Accuracy:", total_accuracy)
    print("Class-wise Accuracy:", class_wise_accuracy)

# q1()
# q2()
q3()