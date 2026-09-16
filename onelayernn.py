#2026 made by Jonathan Farkas
import numpy as np

rng = np.random.default_rng()

#initialize all variables
numOfNeurons = 3
weights = [rng.standard_normal() for x in range(numOfNeurons - 1)]
biases = [rng.standard_normal() for z in range(numOfNeurons - 1)]
inputValue = float(input("Enter the networks starting activation: "))
desiredOutput = float(input("Enter the networks desired output (between 0 and 1): "))
learningRate = 0.01


def sigmoidFunction(x):
    return 1 / (1 + np.exp(-x))

def sigmoidDerivitive(x):
    return sigmoidFunction(x)*(1-sigmoidFunction(x))

def forwardProp():
    activations = [inputValue]
    #loop through the num of neurons to calculate the activations for each, and pass forward
    for x in range(numOfNeurons - 1):
        previousActivation = activations[x]
        activations.append(sigmoidFunction(weights[x]*previousActivation+biases[x]))

    #return the last nuron activation as the systems output
    return activations

def backProp(numOfEpochs):

    #loop through each nuron (minus the input nuron b/c there is no weight and bias that needs to be updated there)
    #will run in backwards order
    for i in range(numOfEpochs):
        activations = forwardProp()
        derivativeToCost = activations[-1] - desiredOutput
        for x in reversed( range(len(activations))):
            #will run 2,1,0
            if x == 0:
                break

            #atp, the derivative to cost represents the last nurons activation to the cost, so we need to take the derivative
            #of the sigmoid with respect to the cost
            derivativeToCost = derivativeToCost * (activations[x]*(1-activations[x]))

            #atp, derivative to cost represents the gradient for the bias
            #print("Old bias: " + str(biases[x-1]))

            #updates bias
            biases[x - 1] = biases[x - 1] - (learningRate*derivativeToCost)
            #print("New bias: " + str(biases[x - 1]))

            #find derivative to cost for weight
            derivativeToCostWrtWeight = derivativeToCost * activations[x-1]
            #update new weight value
            oldWeight = weights[x-1]
            weights[x-1] = weights[x-1] - (learningRate * derivativeToCostWrtWeight)

            #update derivativeToCost WRT the next activation
            derivativeToCost = derivativeToCost * oldWeight

    #work backwards through all of the activations and update the corresponding weights and biases

def calculateCost():
    outputActivation = forwardProp()[-1]
    return 0.5 * (outputActivation-desiredOutput) ** 2

def showWeightsAndBiases():
    for x in weights:
        print("Weight: " + str(x))
    print("\n")
    for x in biases:
        print("Bias: " + str(x))

def showCommandMenu():
    print("1- Train model")
    print("2- Calculate Cost")
    print("3- Forward Prop")
    print("4- See weights/biases")
    print("5- Quit")



while True:
    showCommandMenu()
    userInput = int(input("Enter your input: "))
    if userInput == 1:
        numOfEpoch = int(input("How many training cycles would you like to run: "))
        backProp(numOfEpoch)
        print("Model trained " + str(numOfEpoch) + " times")
    elif userInput == 2:
        print("Cost is " + str(calculateCost()))
    elif userInput == 3:
        print("Forward prop result is " + str(forwardProp()[-1]))
    elif userInput == 4:
        showWeightsAndBiases()
    elif userInput == 5:
        break