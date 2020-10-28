import numpy as np
from sklearn.metrics import confusion_matrix

class Knn:
    """Classifier implementing the k-nearest neighbors vote.

    Parameters
    __________
    :param n_neighbors : int
        Number of neighbors to use.

    :param metric : {'manhattan', 'euclidean', 'minkowski'}, default='minkowski'
        The distance metric to use for defining K-nearest neighbors. The default metric is minkowski, and with p=2 is
        equivalent to the standard Euclidean metric.

    :param p : int, default=2
        Power parameter for the Minkowski metric. When p=1, this is equivalent to using manhattan_distance (l1), and
        euclidean_distance (l2) for p=2.

    :param weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood are weighted equally.
        - 'distance' : weight points by the inverse of their distance. In this case, closer neighbors of a query point
        will have a greater influence than neighbors which are further away.

    Methods
    __________
    :method fit :
        It fits the model using X as training data and y as target values.

    :method predict :
        Loop through all data points and predict the class labels for each of the new data point based on training data.
    """

    def __init__(self, n_neighbors, metric='minkowski', p=2, weights='uniform'):

        if p < 0:
            raise ValueError("p should be larger than 0.")

        if metric not in ['minkowski', 'manhattan', 'euclidean']:
            raise ValueError("Distance method not supported. Must be {'manhattan', 'euclidean', 'minkowski'}")

        if weights not in ['uniform', 'distance']:
            raise ValueError(
                "Weights can be only assigned uniformly or based on distance. Must be {'uniform', 'distance'}")

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.weights = weights

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values
        Parameters
        __________
        :argument X:   {array-like, sparse matrix}
                    Training data. If array or matrix, shape = [n_samples, n_features]

        :argument y:   {array-like, sparse matrix}
                    Target values of shape = [n_samples] or [n_samples, n_outputs]

        :return:    Training data and associated labels
        """

        # check data shape
        if X.shape[0] == y.shape[0]:
            self.X = X
            self.y = y
        else:
            raise ValueError("Dimensional mismatch: Number of rows in X must be equal to the number of rows in y")

        # check for missing values
        if np.isnan(X).any() or np.isnan(y).any():
            raise TypeError("There are missing values in the dataset. Consider removing samples with missing values"
                            "or imputation methods.")
        return X, y

    def _manhattan_distance(self, point):
        """Calculate manhattan distance from one data point to all the samples in the training set.

        :param point: {array-like}
            New data point of shape [n_features]

        :return: numpy array with manhattan distances from the data point to all the samples in the training set.
        """
        return np.sum(abs(self.X - point), axis=1)

    def _euclidean_distance(self, point):
        """Calculate euclidean distance from one data point to all the samples in the training set.

        :param point: {array-like}
            New data point of shape [n_features]

        :return: numpy array with euclidean distances from the data point to all the samples in the training set.
        """
        return np.sqrt(np.sum((self.X - point) ** 2, axis=1))

    def _minkowski_distance(self, point):
        """Calculate minkowski distance from one data point to all the samples in the training set.

        :param point: {array-like}
            New data point of shape [n_features]

        :return: numpy array with minkowski distances from the data point to all the samples in the training set.
        """
        return np.sum(abs(self.X - point) ** self.p, axis=1) ** (1 / self.p)

    def _uniform_weights(self, distances):
        """Assign equal weights to all points.

        :param distances: {array-like}
            numpy array with distances from one data point to all the samples in the training set.

        :return: numpy array with weight-distance pairs for each sample in the training set.
        """
        return np.array([(1, d) for _, d in enumerate(distances)])

    def _distance_weights(self, distances):
        """Weight points by the inverse of their distance.

        :param distances: {array-like}
            numpy array with distances from one data point to all the samples in the training set.

        :return: numpy array with weight-distance pairs for each sample in the training set.
        """
        return np.array([(1 / d, d) if d > 0 else (1, d) for _, d in enumerate(distances)])

    def _predict_point(self, point):
        """ Predict class label of a single data point.

        :argument point: {array-like}
            New data point of shape [n_features]
        :return: str
            Assigned class label based on training data.
        """
        # calculate point distance from all other samples
        if self.metric == 'manhattan':
            distances = self._manhattan_distance(point)
        elif self.metric == 'euclidean':
            distances = self._euclidean_distance(point)
        elif self.metric == 'minkowski':
            distances = self._minkowski_distance(point)
        else:
            AttributeError("Distance method not supported. Must be {'manhattan', 'euclidean', 'minkowski'}")

        # calculate point distance weights
        if self.weights == 'uniform':
            weights = self._uniform_weights(distances)
        else:
            weights = self._distance_weights(distances)

        # sort index of distances from nearest to farthest and keep only first "n_neighbors" ones
        sorted_distances_idxs = distances.argsort()[:self.n_neighbors]

        # Vote - count number of classes for Knn
        class_count = {}

        if self.weights == 'uniform':
            # assign uniform weights
            for idx in sorted_distances_idxs:
                vote_label = self.y[idx]
                class_count[vote_label] = class_count.get(vote_label, 0) + 1
        else:
            # assign weights based on distance
            for idx in sorted_distances_idxs:
                vote_label = self.y[idx]
                class_count[vote_label] = class_count.get(vote_label, 0) + weights[idx][0]

        # Descending sort the resulting class counts dictionary by class counts values
        sorted_class_count = sorted(class_count.items(),
                                    key=lambda item: (item[1], item[0]),
                                    reverse=True)

        # Return the predicted label
        return sorted_class_count[0][0]

    def predict(self, x):
        """Loop through all data points and predict the class labels

        :argument x: {array-like}
             New data points to be assigned a label of shape [n_points, n_features]

        :return: list
            A list with class labels assign to each new data point.
        """
        # Loop through all samples and predict the class labels and store the results
        return [self._predict_point(point) for point in x]

    def display_results(self, y_test, y_pred):
        labels = np.unique(y_pred)
        confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
        accuracy = (y_pred == y_test).mean()

        print("Labels:", labels)
        print("Confusion Matrix:\n", confusion_mat)
        print("Accuracy:", accuracy)

    def __repr__(self):
        return "<n_neighbors:"+self.n_neighbors+", metric:" +self.metric+", p:"+str(self.p)+", weights:"+self.weights+">"

    def __str__(self):
        return "Knn(n_neighbors="+self.n_neighbors+", metric=" +self.metric+", p="+str(self.p)+", weights="+self.weights+")"
