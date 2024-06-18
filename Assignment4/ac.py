import numpy as np
import matplotlib.pyplot as plt

class AdaBoostM1:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None
        self.alphas = None
        self.stumps = None
        self.val_accuracies = None

    def load_data(self, path):
        with np.load(path) as f:
            self.x_train, self.y_train = f['x_train'], f['y_train']
            self.x_test, self.y_test = f['x_test'], f['y_test']

    def filter_data(self):
        idx = np.where((self.y_train == 0) | (self.y_train == 1))
        self.x_train, self.y_train = self.x_train[idx], self.y_train[idx]

    def split_data(self, val_size=1000):
        idx_0 = np.where(self.y_train == 0)[0]
        idx_1 = np.where(self.y_train == 1)[0]
        self.x_val = np.concatenate((self.x_train[idx_0[:val_size]], self.x_train[idx_1[:val_size]]))
        self.y_val = np.concatenate(([-1]*val_size, self.y_train[idx_1[:val_size]]))
        self.x_train = np.concatenate((self.x_train[idx_0[val_size:]], self.x_train[idx_1[val_size:]]))
        self.y_train = np.concatenate(([-1]*(len(idx_0)-val_size), self.y_train[idx_1[val_size:]]))

    def apply_pca(self, x, n_components=5):
        x_flat = x.reshape(x.shape[0], -1)
        cov_matrix = np.cov(x_flat.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        top_indices = sorted_indices[:n_components]
        pca_matrix = eigenvectors[:, top_indices]
        return np.dot(x_flat, pca_matrix)

    def train_decision_stump(self, x, y, weights):
        best_dim = 0
        best_thresh = 0
        min_error = np.inf
        
        for dim in range(x.shape[1]):
            unique_values = np.unique(x[:, dim])
            for i in range(len(unique_values) - 1):
                thresh = (unique_values[i] + unique_values[i+1]) / 2
                predictions = np.sign(x[:, dim] - thresh)
                error = np.sum(weights * (predictions != y))
                if error < min_error:
                    min_error = error
                    best_dim = dim
                    best_thresh = thresh
                    
        return best_dim, best_thresh

    def update_weights(self, weights, alpha, predictions, y):
        new_weights = weights * np.exp(-alpha * y * predictions)
        return new_weights / np.sum(new_weights)

    def predict_stump(self, x, dim, thresh):
        return np.sign(x[:, dim] - thresh)

    def compute_accuracy(self, predictions, y):
        return np.mean(predictions == y)

    def initialize_weights(self, n):
        return np.ones(n) / n

    def adaboost_m1(self, num_iterations = 5):
        n_train = len(self.x_train)
        n_val = len(self.x_val)
        weights = self.initialize_weights(n_train)
        self.alphas = []
        self.stumps = []
        self.val_accuracies = []
        
        for t in range(num_iterations):
            # Train decision stump
            dim, thresh = self.train_decision_stump(self.x_train, self.y_train, weights)
            self.stumps.append((dim, thresh))
            
            # Predictions
            predictions = self.predict_stump(self.x_train, dim, thresh)
            
            # Compute weighted error
            weighted_error = np.sum(weights * (predictions != self.y_train))

            # Compute alpha
            alpha = 0.5 * np.log((1 - weighted_error) / weighted_error)
            self.alphas.append(alpha)

            # Update weights
            weights = self.update_weights(weights, alpha, predictions, self.y_train)

            # Evaluate on validation set
            val_predictions = np.zeros(n_train)
            for i in range(len(self.stumps)):
                dim, thresh = self.stumps[i]
                val_predictions += self.alphas[i] * self.predict_stump(self.x_train, dim, thresh)
            val_accuracy = self.compute_accuracy(np.sign(val_predictions), self.y_train)
            self.val_accuracies.append(val_accuracy)
            print(f"Iteration {t+1}, Validation Accuracy: {val_accuracy:.4f}")

    def plot_accuracy(self):
        plt.plot(range(1, len(self.val_accuracies) + 1), self.val_accuracies)
        plt.xlabel('Number of Trees')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy vs. Number of Trees')
        plt.show()

    def evaluate_test(self):
        test_predictions = np.zeros(len(self.x_val))
        for i in range(len(self.stumps)):
            dim, thresh = self.stumps[i]
            test_predictions += self.alphas[i] * self.predict_stump(self.x_val, dim, thresh)
        test_accuracy = self.compute_accuracy(np.sign(test_predictions), self.y_val)
        print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    adaboost = AdaBoostM1()
    adaboost.load_data("C:\Shared_archcraft\SML\Assignment4\mnist.npz")
    adaboost.filter_data()
    adaboost.split_data()
    adaboost.x_train = adaboost.apply_pca(adaboost.x_train)
    adaboost.x_val = adaboost.apply_pca(adaboost.x_val)
    adaboost.adaboost_m1()
    adaboost.plot_accuracy()
    adaboost.evaluate_test()
