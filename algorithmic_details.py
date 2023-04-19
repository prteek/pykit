import pykitml as pk
from pykitml.datasets import fishlength
# Load the dataset
inputs, outputs = fishlength.load()

# Normalize inputs
array_min, array_max = pk.get_minmax(inputs)
inputs = pk.normalize_minmax(inputs, array_min, array_max)
optimizer = pk.GradientDescent(learning_rate=0.01, decay_rate=0)

# ------ Setup model ------ #
import numpy as np
import tqdm

self = lambda x:x
input_size = inputs.shape[1]
output_size = 1
reg_param = 0

# Save sizes
self._input_size = input_size
self._output_size = output_size

# Initialize regularization parameter
self._reg_param = reg_param
self._reg_param_half = reg_param/2

# Initialize weights and parameters
epsilon = np.sqrt(6)/(np.sqrt(output_size) + np.sqrt(input_size))
weights = np.random.rand(output_size, input_size)*2*epsilon - epsilon
biases = np.random.rand(output_size) * 2 * epsilon - epsilon

self._inputa = np.array([])
self.a = np.array([])
self.z = np.array([])

# Put parameters in numpy dtype=object array
W = 0  # Weights
B = 1  # Biases
self._params = np.array([None, None], dtype=object)
self._params[W] = weights
self._params[B] = biases

self._mparams = self._params
# Activations refer to the value of the function itself before weight update
self._activ_func = lambda weighted_sum: weighted_sum  # Identity function as activation
self._activ_func_prime = lambda weighted_sum, activations: 1  # the derivative of identity w.r.t layer's weighted sum.
self._cost_func = lambda output, target: 0.5 * ((output - target) ** 2)
self._cost_func_prime = lambda output, target: output - target

def _backpropagate(self, index, target):
    # Constants
    W = 0  # Weights
    B = 1  # Biases

    # Gradients
    dz_dw = self._inputa[index]
    da_dz = self._activ_func_prime(self.z[index], self.a[index])
    dc_db = self._cost_func_prime(self.a[index], target) * da_dz
    dc_dw = np.multiply.outer(dc_db, dz_dw)

    # Add regularization
    dc_dw += self._reg_param*self._params[W]

    # Return gradient
    gradient = np.array([None, None], dtype=object)
    gradient[W] = dc_dw
    gradient[B] = dc_db
    return gradient

self._backpropagate = lambda *args: _backpropagate(self, *args)

def feed(self, input_data):
    # Constants
    W = 0  # Weights
    B = 1  # Biases

    # feed
    self._inputa = input_data
    self.z = (input_data @ self._params[W].T) + self._params[B]
    self.a = self._activ_func(self.z)

self.feed = lambda *args: feed(self, *args)


def _get_norm_weights(self):
    W = 0
    return self._reg_param_half*(self._params[W]**2).sum()

self._get_norm_weights = lambda *args: _get_norm_weights(self, *args)



# ------ Train model ------ #
training_data=inputs
targets=outputs
batch_size=20
epochs=200
optimizer=pk.GradientDescent(learning_rate=0.01, decay_rate=0)
testing_data=None
testing_targets=None
testing_freq=1
decay_freq=1
self.bptt = False

self._performance_log = {}
self._performance_log['epoch'] = []
self._performance_log['cost_train'] = []
self._performance_log['learning_rate'] = []

# if testing_data is not None:
#     self._performance_log['cost_test'] = []
# self._init_train(batch_size)
def _get_batch_grad(self, epoch, batch_size, training_data, targets):
    # Total gradient for the batch
    total_gradient = 0  # zero grad pytorch

    # feed the batch
    start_index = (epoch*batch_size) % training_data.shape[0]
    end_index = start_index+batch_size
    indices = np.arange(start_index, end_index) % training_data.shape[0]
    self.feed(training_data[indices])

    if self.bptt:
        return self._backpropagate(None, targets[indices])

    # Loop through the batch
    for example in range(0, batch_size):
        # Add the calculated gradients to the total
        index = ((epoch*batch_size) + example) % training_data.shape[0]
        # Get gradient
        total_gradient += self._backpropagate(example, targets[index])

    # return the total divide by batch size to not make it explode.
    # Dividing essentially acts together with learning_rate later on
    return total_gradient/batch_size

self._get_batch_grad = lambda *args: _get_batch_grad(self, *args)


pbar = tqdm.trange(0, epochs, ncols=80, unit='epochs')
for epoch in pbar:
    total_gradient = self._get_batch_grad(epoch, batch_size, training_data, targets)

    # After completing a batch, average the total sum of gradients and tweak the parameters
    self._mparams = optimizer._optimize(self._mparams, total_gradient)
    # Zeros your gradients for every batch: optimizer.zero_grad()
    # Makes predictions for this batch: outputs = model(inputs)
    # Compute the loss and its gradients: loss = loss_fn(outputs, labels); loss.backward()
    # Adjust learning weights: optimizer.step()

    # Decay the learning rate (not a part of pytorch)
    if (epoch+1) % decay_freq == 0:
        optimizer._decay()

    # Log and print performance
    if (epoch+1) % testing_freq == 0:
        # log epoch
        self._performance_log['epoch'].append(epoch+1)

        # log learning rate
        learning_rate = optimizer._learning_rate
        self._performance_log['learning_rate'].append(learning_rate)

        # get cost of the model on training data
        cost_train = self.cost(training_data, targets)
        pbar.set_postfix(cost=cost_train)
        self._performance_log['cost_train'].append(cost_train)

