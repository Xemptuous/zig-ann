# zig-ann
An example of using Zig to create a simple ANN with 1 hidden layer simulating an XOR gate.

`Activation Function` - Hyperbolic Tangent (tanh)

`Loss Function` - Sum of Squared Errors  (SSE)

## Adjusting

Global variables at the top of `main.zig` can be used to tune the ANN.

`BIAS`
: The global bias for each neuron.

`ALPHA`
: The "learning rate"; the smaller the number, the more accurate the results, but the longer it takes to train.

`NUM_HIDDEN`
: The number of hidden neurons in the hidden layer.

`NUM_INPUTS`
: The number of inputs fed to the neurons in the hidden layer.

`NUM_OUTPUTS`
: The expected number of outputs in the output layer.

`NUM_TEST_CASES`
: The number of test cases being fed to the model to be used in each epoch.

`ERROR_THRESHOLD`
: The threshold of the loss function that signals a successful training and an exiting of the function.

`EPOCH_THRESHOLD`
: The number of epochs before restarting and rerolling of weights occurs.

`EPOCH_PRINT_THRESHOLD`
: The number of epochs that must pass before printing status updates.
