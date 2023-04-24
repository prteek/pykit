import pykitml as pk
from pykitml.datasets import fishlength
import matplotlib.pyplot as plt


class PKtoSKAdapter:
    def __init__(self, model, **train_kwargs):
        self._model = model
        self.train_kwargs = train_kwargs

    def fit(self, X, y):
        self._model.train(X, y, **self.train_kwargs)

    def predict(self, X):
        self._model.feed(X)
        return self._model.get_output()

    def set_params(self, **kwargs):
        self.train_kwargs.update(**kwargs)
        return self.train_kwargs

    def __getattr__(self, attr):
        """All non-adapted calls are passed to the object"""
        if attr == "_model":  # Prevent infinite recursion on model load
            pass
        else:
            return getattr(self._model, attr)


# Load the dataset
inputs, outputs = fishlength.load()

# Normalize inputs
array_min, array_max = pk.get_minmax(inputs)
inputs = pk.normalize_minmax(inputs, array_min, array_max)

# Create polynomial features
inputs_poly = pk.polynomial(inputs)

# Normalize outputs
array_min, array_max = pk.get_minmax(outputs)
outputs = pk.normalize_minmax(outputs, array_min, array_max)

# Create model
fish_classifier = pk.LinearRegression(inputs_poly.shape[1], 1)

opt = pk.Momentum(learning_rate=0.2)

model = PKtoSKAdapter(fish_classifier,
                      batch_size=22,
                      epochs=2000,
                      optimizer=opt,
                      testing_freq=1,
                      decay_freq=10)

# Train the model
model.fit(inputs_poly, outputs)

# Plot performance
model.plot_performance()

# Print r2 score
print('r2score:', model.r2score(inputs_poly, outputs))
plt.scatter(outputs, model.predict(inputs_poly))
plt.show()

# Save model
pk.save(model, 'fish_classifier.pkl')

mod = pk.load('fish_classifier.pkl')

mod.predict(inputs_poly)
