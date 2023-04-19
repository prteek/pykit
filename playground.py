import pykitml as pk
from pykitml.datasets import fishlength

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

# Train the model
fish_classifier.train(
    training_data=inputs_poly,
    targets=outputs,
    batch_size=22,
    epochs=500,
    optimizer=pk.GradientDescent(learning_rate=0.01, decay_rate=0.9),
    testing_freq=1,
    decay_freq=10
)

# Save model
pk.save(fish_classifier, 'fish_classifier.pkl')

# Plot performance
fish_classifier.plot_performance()

# Print r2 score
print('r2score:', fish_classifier.r2score(inputs_poly, outputs))

