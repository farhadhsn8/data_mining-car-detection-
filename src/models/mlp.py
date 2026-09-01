from sklearn.neural_network import MLPClassifier


class MLPModel:
    def __init__(self, solver="lbfgs", alpha=0.1, hidden_layer_sizes=(4, 2), random_state=1):
        self.model = MLPClassifier(
            solver=solver,
            alpha=alpha,
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
