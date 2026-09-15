# Single Layer Neural Network

## Description:
The neural network has the following behaviour: A1 -> A2 -> A3... Ax

All weights and biases are initialized randomly, and all that has to be entered by the user is the input activation, the desired output 
and the number of perceptrons in the network and how many training epochs they want to do. After each epoch, you can print the cost 
of the network, modeled by the function c = 1/2(y-Ax)^2 where c is the cost, y is the desired output, and Ax is the output the network gave. 

When training, the network backpropgates and uses gradient descent to auto update the weights and biases of each neuron to minimize the cost according to its 
specified learning rate.  

After training, simply running a forward propagation pass will show the intelligence that has been baked into the weights and biases of this very simple
network.
