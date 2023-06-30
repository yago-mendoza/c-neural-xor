#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/////////////////////////////
// Preprocessing Directives
/////////////////////////////

// Neurons
#define numInputs 2
#define numHiddenNodes 2
#define numHiddenLayers 1
#define numOutputs 1
// Training data
#define numTrainingSets 4
// Other parameters
#define learningRate 0.1f
#define numEpochs 10000

// Function Prototypes

double sigmoid(double x) {return 1 / (1 + exp(-x));}
double dSigmoid(double x) {return x * (1 - x);}

double init_weights() {return ((double)rand()) / ((double)RAND_MAX);}

void shuffle(int *array, size_t n) {
    // Fisher-Yates shuffle algorithm
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

//////////////////////////////////////////////////////////
// MAIN FUNCTION
//////////////////////////////////////////////////////////

int main(void) {

    // Architecture (values, biases and weights)

    //// Hidden layer
    double hiddenLayer[numHiddenLayers][numHiddenNodes];
    double hiddenLayerBias[numHiddenLayers][numHiddenNodes];

    //// Input layer to first hidden layer weights
    double inputWeights[numInputs][numHiddenNodes];
    for(int i = 0; i < numInputs; i++) {
        for(int j = 0; j < numHiddenNodes; j++) {
            inputWeights[i][j] = init_weights();
        }
    }

    //// Weights between hidden layers
    double hiddenWeights[numHiddenLayers - 1][numHiddenNodes][numHiddenNodes];
    for(int l = 0; l < numHiddenLayers - 1; l++) {
        for(int i = 0; i < numHiddenNodes; i++) {
            for(int j = 0; j < numHiddenNodes; j++) {
                hiddenWeights[l][i][j] = init_weights();
            }
        }
    }

    //// Output layer
    double outputLayer[numOutputs];
    double outputLayerBias[numOutputs]; 
    double outputWeights[numHiddenNodes][numOutputs];

    for(int i = 0; i < numOutputs; i++) {
        outputLayerBias[i] = init_weights();
        for(int j = 0; j < numHiddenNodes; j++) {
            outputWeights[j][i] = init_weights();
        }
    }

    // Training data extraction

    int trainingSetOrder[numTrainingSets]; // {0, 1, 2, 3, ...}
    for (int i = 0; i < numTrainingSets; i++) {
        trainingSetOrder[i] = i;
    }

    FILE *fp;
    char str[60];
    char *filename = "training_data.txt";
    fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Could not open file %s", filename);
        return 1;
    }

    double trainingInputs[numTrainingSets][numInputs];
    double trainingOutputs[numTrainingSets][numOutputs];

    int i = 0;
    while (fgets(str, 60, fp) != NULL) {
        char *token = strtok(str, " ");
        int j = 0;
        while (token != NULL) {
            if (j < numInputs) {
                trainingInputs[i][j] = atof(token);
            } else {
                trainingOutputs[i][0] = atof(token);
                break;
            }
            token = strtok(NULL, " "); // Move to the next token
            j++;
        }
        i++;
    }

    fclose(fp);

    printf("Training data: \n"); 
    for(int i = 0; i < numTrainingSets; i++) {
        printf("Training set %d, Inputs: ", i);
        for(int j = 0; j < numInputs; j++) {
            printf("%g ", trainingInputs[i][j]);
        }
        printf(", Expected Outputs: ");
        for(int k = 0; k < numOutputs; k++) {
            printf("%g ", trainingOutputs[i][k]);
        }
        printf("\n");
    }

    printf("> Press ENTER to continue...");
    getchar();

    /////////////////////////////
    // Training
    /////////////////////////////

    for (int epoch = 0; epoch < numEpochs; epoch++) {
        shuffle(trainingSetOrder, numTrainingSets);

        for (int i = 0; i < numTrainingSets; i++) {
            // 1. Forward-propagation

            // 1.1. Calculates the values of the first hidden layer nodes
            for(int j = 0; j < numHiddenNodes; j++) {
                double activation = hiddenLayerBias[0][j];
                for(int k = 0; k < numInputs; k++) {
                    activation += trainingInputs[i][k] * inputWeights[k][j];
                }
                hiddenLayer[0][j] = sigmoid(activation);
            }

            // 1.2. Calculates the values of the remaining hidden layers nodes
            for(int l = 1; l < numHiddenLayers; l++) {
                for(int j = 0; j < numHiddenNodes; j++) {
                    double activation = hiddenLayerBias[l][j];
                    for(int k = 0; k < numHiddenNodes; k++) {
                        activation += hiddenLayer[l-1][k] * hiddenWeights[l-1][k][j];
                    }
                    hiddenLayer[l][j] = sigmoid(activation);
                }
            }

            // 1.3. Calculates the value of the output layer node
            for(int j = 0; j < numOutputs; j++) {
                double activation = outputLayerBias[j];
                for(int k = 0; k < numHiddenNodes; k++) {
                    activation += hiddenLayer[numHiddenLayers-1][k] * outputWeights[k][j];
                }
                outputLayer[j] = sigmoid(activation);
            }

            // Display current evolution
            printf("Epoch: %d, Input: (", epoch);
            for (int j = 0; j < numInputs; j++) {
                printf("%g", trainingInputs[i][j]);
                if (j != numInputs - 1) {
                    printf(", ");
                }
            }
            printf("), Expected Output: %g, Current Output: %g\n", trainingOutputs[i][0], outputLayer[0]);

            // 2. Back-propagation

            // 2.1. Calculates the error of the output layer node
            double deltaOutput[numOutputs];
            for(int j = 0; j < numOutputs; j++) {
                double error = (trainingOutputs[i][j] - outputLayer[j]);
                deltaOutput[j] = error * dSigmoid(outputLayer[j]);
            }

            // 2.2. Calculates the error of the hidden layer nodes
            double deltaHidden[numHiddenLayers][numHiddenNodes];
            for(int l = numHiddenLayers-1; l >= 0; l--) {
                for(int j = 0; j < numHiddenNodes; j++) {
                    double error = 0;
                    if(l == numHiddenLayers-1) {
                        for(int k = 0; k < numOutputs; k++) {
                            error += deltaOutput[k] * outputWeights[j][k];
                        }
                    } else {
                        for(int k = 0; k < numHiddenNodes; k++) {
                            error += deltaHidden[l+1][k] * hiddenWeights[l][j][k];
                        }
                    }
                    deltaHidden[l][j] = error * dSigmoid(hiddenLayer[l][j]);
                }
            }

            // 3. Updates weights

            // 3.1. Updates the weights of the output layer node
            for(int j = 0; j < numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j] * learningRate;
                for(int k = 0; k < numHiddenNodes; k++) {
                    outputWeights[k][j] += hiddenLayer[numHiddenLayers-1][k] * deltaOutput[j] * learningRate;
                }
            }

            // 3.2. Updates the weights of the hidden layer nodes
            for(int l = 0; l < numHiddenLayers; l++) {
                for(int j = 0; j < numHiddenNodes; j++) {
                    hiddenLayerBias[l][j] += deltaHidden[l][j] * learningRate;
                    if(l == 0) {
                        for(int k = 0; k < numInputs; k++) {
                            inputWeights[k][j] += trainingInputs[i][k] * deltaHidden[l][j] * learningRate;
                        }
                    } else {
                        for(int k = 0; k < numHiddenNodes; k++) {
                            hiddenWeights[l-1][k][j] += hiddenLayer[l-1][k] * deltaHidden[l][j] * learningRate;
                        }
                    }
                }
            }
        }
    }

    // Final results
    printf("\nFinal weights and biases after training:\n");

    // Input layer weights
    printf("\nInput layer weights:\n");
    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numHiddenNodes; j++) {
            printf("Weight[%d][%d]: %f\n", i, j, inputWeights[i][j]);
        }
    }
    printf("\n");

    // Hidden layer weights
    printf("Inter-hidden layer weights:\n");
    for (int l = 0; l < numHiddenLayers-1; l++) {
        printf("Hidden Layer %d:\n", l+1);
        for (int i = 0; i < numHiddenNodes; i++) {
            for (int j = 0; j < numHiddenNodes; j++) {
                printf("Weight[%d][%d][%d]: %f\n", l, i, j, hiddenWeights[l][i][j]);
            }
        }
        printf("\n");
    }

    // Hidden layer biases
    printf("Hidden layer biases:\n");
    for (int l = 0; l < numHiddenLayers; l++) {
        printf("Hidden Layer %d:\n", l+1);
        for (int i = 0; i < numHiddenNodes; i++) {
            printf("Bias[%d][%d]: %f\n", l, i, hiddenLayerBias[l][i]);
        }
        printf("\n");
    }

    // Output layer weights
    printf("Output layer weights:\n");
    for (int j = 0; j < numOutputs; j++) {
        printf("Output Node %d:\n", j+1);
        for (int k = 0; k < numHiddenNodes; k++) {
            printf("Weight[%d][%d]: %f\n", k, j, outputWeights[k][j]);
        }
        printf("\n");
    }

    // Output layer biases
    printf("Output layer biases:\n");
    for (int i = 0; i < numOutputs; i++) {
        printf("Bias[%d]: %f\n", i, outputLayerBias[i]);
    }

    return 0;
}