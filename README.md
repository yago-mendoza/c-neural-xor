# c-neural-xor

## General Information

This project aims to consolidate and operationalize a fundamental XOR neural network with backpropagation. It involves implementing the network structure in C, training it using backpropagation, and testing its ability to solve the XOR logical operation. By demonstrating the practical application and functioning of neural networks and their training process on a computer platform, this project provides insight into the inner workings of the network.

## Running the NN

Compile the NN.c file using the GCC compiler by running the command:

```bash
gcc -o start NN.c -lm
```

Once the compilation is successful, execute the program by running the command: 

```bash
./start.exe
```

## Personalizing training data

To modify the input data, edit the TXT file following this format: each row represents a training set with input data separated by spaces, followed by the desired output value. Multiple training sets can be defined within the file. Remember to update the preprocessing directives in the file if necessary.
