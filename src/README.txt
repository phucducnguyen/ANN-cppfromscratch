TRAINING:
for Input:
we need struct_training.txt and two training files, which you can set name in the struct file
struct_training.txt :  is used to set up Training section.
	input_trainingfileName: 
	output_trainingfileName: 
	NumOfIter 
	LearningRate 
	NumOfInputs 
	NumOfOutputs 
	NumOfSamples 
	NumOfHiddenLayers 
	NumOfNeuronsPerHiddenLayer 
for Output:
The program will create directories (if it's not existed), based on the name of the struct file
	for example: I2H2N3N3O1 -2 inputs
				-2 hidden layers (3 neurons for 1st layer,3 neurons for 2nd layer)
				-1 output
InterimResults.txt :	show intermediate process
error.txt : 		show the error improvement of learning process
out.xls : 		error improvement of learning process for graph purpose
trained_weight.txt :	show the final weight after the learning process
trained_bias.txt :	show the final bias after the learning process
normalize.txt :		store normalized paramenter of the input
min_weight.txt :	show the minimum weight after the learning process
min_bias.txt :		show the minimum bias after the learning process
min_J.txt :		store Iter position in which min error appear

EXECUTING:
for Input:
we need struct_executing.txt and normalize.txt
struct_training.txt :  is used to set up Training section.
	weight_fileName: set weigh input file
	bias_fileName:   set bias input file	
	input_fileName:  set input data file for executing process
	NumOfInputs 
	NumOfOutputs
	NumOfSamples
	NumOfHiddenLayers
	NumOfNeuronsPerHiddenLayer 

for Output:
The program will create directories (if it's not existed), based on the name of the struct file
	for example: I2H2N3N3O1 -2 inputs
				-2 hidden layers (3 neurons for 1st layer,3 neurons for 2nd layer)
				-1 output
InterimResults.txt : show intermediate process
trained_weight.txt : show the weights that are used in the process
trained_bias.txt : show the bias that are used in the process
normalize.txt :  show normalized parameters are used the input
