from sklearn import linear_model


class LinearRegressionModel:
    """Multi-output linear regression used on the one-hot encoded targets."""

    def __init__(self):
        self.model = linear_model.LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def score(self, X, y):
        return self.model.score(X, y)
