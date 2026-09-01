from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    def __init__(self, random_state=0):
        self.model = LogisticRegression(random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
