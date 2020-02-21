#include "NeuralNetwork.h"
#include <random>

int main(int argc, char* argv[])//int argc, char* argv[]
{	
	std::srand(std::time(NULL));
	int i,j;
	//float learningRate=0.01;//0.0005;//0.01;
	//int iteration=3;
	int randWeight=0;
	int setWeight=0;
	if (argc<2){
		randWeight=1;//A.RandomizeWeights();
	}
	else if (strcmp(argv[1],"--help")==0){
		std::cerr << "Usage: \t-r: repeat weight by input weights and bias\n\t-r <input_weight.txt> <input_bias.txt>\n\tor -r" << std::endl;
		return 0;
	}
	else if(strcmp(argv[1],"-r")==0){
		setWeight=1;//A.SetWeights(argv[2],argv[3]);
	}
	
	
	NeuralNetwork A("struct_training.txt");
	A.InputOutputTraining();//"training_input.txt","training_output.txt");
	//A.InputOutputSample("trainingdata.txt","trainingdata_out.txt");
	//A.SetWeightsBias("trained_weight.txt","trained_bias.txt");
	if (setWeight==1){
		A.SetWeightsBias(argv[2],argv[3]);
	}
	if (randWeight==1){
		A.RandomizeWeights();
	}
	A.Training();//iteration,learningRate);
	return 0;
}
