/*Supervised Deep Learning*/
//Linear regression with Gradient descent
//ADD -lstdc++fs at the end
#pragma once
#include "MatrixVector.h"
#include <experimental/filesystem>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <string.h>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <limits>

namespace fs=std::experimental::filesystem;
/*
ifstream input_file_p0; // declare a pointer to an input file;
input_file_p0.open("studentId.txt", ios::in); // open input file;
input_file.close();
input_file.clear();
 */
//Declared output file in constructor
std::ofstream output_excelfile;
std::ofstream cost_output;
//std::ofstream InterimResults("InterimResults.txt", std::ios::out);//std::ios::app);
std::ofstream InterimResults;



#define LOG_START InterimResults << "\t*** Process Begins ***" << std::endl
#define LOG_END InterimResults << "\t*** Process Ends ***" << std::endl

class NeuralNetwork {
    public: //public interface for this class
		NeuralNetwork (){}; //Constructor
        NeuralNetwork (const char* filename,int execute); //Constructor
		//~NeuralNetwork(){InterimResults.close();}
        void InputOutputTraining ();//const char* filenameIn,const char* filenameOut);
        void RandomizeWeights();
		void SetWeightsBias();
		void SetWeightsBias(const char * weights_file, const char * bias_file);
		void Normalized();
		void Normalized(std::vector<float> v1,std::vector<float> v2);
		void Denormalized();
		void Denormalized(std::vector<float> v1,std::vector<float> v2);
		void SetInput();//const char * input_file);
		void ForwardPropagation();
        void CostFunction();
		void BackPropagation(float learningRate);
		void ExportWeightBias(const char* weight_file,const char* bias_file);
		void ExpectedOutPut(float sleep, float study);
		void PrintWeight();
		void PrintBias();
		void Training();//int epoch,float learningRate);
		void _checkStruct();
		void _createDirectories(int execute =0);
		void _exportFiles(int execute);
		///////////////////////Get Method////////////////////////////////
		int n_Input(); //return n_Input
		int n_Output();//return n_Output
		Matrix getX();
		Matrix getY();
		///////////////////////Set Method////////////////////////////////
		void setVector(std::vector<float> vector, std::vector<float> k);
		float getJ(){return J;}
		//void operator=(NeuralNetwork nn);
		//void SetInput(const char * input_file);
		//void SetInput(float sleep, float study);
		//void Training(int epoch,float learningRate);
    private:
        // #ofInput,#ofOutput,#ofSamples,#ofHLayers,#ofNeuronsPerHiddenLayer
        int n_Inputs, n_Outputs, n_Samples, n_HLayers;
		std::vector<int> n_NeuronsPerHLayer;
		std::vector<float> n_NeuronsPerLayers; // store # of neurons per layer include input and output layer
        Matrix X; // matrix store inputs	2x2 vector matrix
	    Matrix Y; // matrix store outputs	2x2 vector matrix
		Matrix W; // temp Weight matrix 	2x2 vector matrix
		Matrix B; // temp Bias matrix		2x2 vector matrix
		Matrix H; // temp matrix H=X.W+B	2x2 vector matrix
		Matrix dW;// temp matrix dJ/dW		2x2 vector matrix
		Matrix dB;// temp matrix dJ/dB		2x2 vector matrix
		float J; //cost of function value
		
	    char inputs_fileName[50]; // hold the inputs file name
	    char outputs_fileName[50]; // hold the outputs file name
		float Weights[20][50][50]; 	// 3D array store Weight with hiddenLayer order
		float Bias[20][1][50];		// 3D array store Bias with hiddenLayer order
	    float Z[20][50][50];		// 3D array store HiddenLayer Output with hiddenLayer order
		//
		std::string directory;
		std::string input_trainingfileName;
		std::string output_trainingfileName;
		std::string weight_fileName;
		std::string bias_fileName;
		std::string realInput_fileName;
		int NumOfIter;
		float learningRate;
};
// Linear Rectifier (ReLU) activation function
float ReLU(float x){ 
	if (x>0) return x;
	else return 0;
}
float ReLUprime(float x){
	if (x>0) return 1;
	else return 0;
}
void NeuralNetwork::Training(){//int epoch,float learningRate){
	std::ofstream min_J("min_J.txt", std::ios::out);//directory+"/"+
	output_excelfile.open("out.xls", std::ios::out);//(directory+"/"+"out.xls", std::ios::out);
	cost_output.open("error.txt", std::ios::out);//(directory+"/"+"cost.txt", std::ios::out);
	int i;
	//A.Normalized();
	float min= std::numeric_limits<float>::max();
	for (i=1;i<=NumOfIter;i++){
		InterimResults<<"\n\t\t*****ITERATION: "<<i<<" *****"<<std::endl;
		ForwardPropagation();
		output_excelfile<<i;
		cost_output <<"Iteration "<<i<<" error: ";
		CostFunction();
		BackPropagation(learningRate);
		if(min > getJ()){
			min=getJ();
			min_J<<"Iter: "<<i<<" Minimum Cost: "<<min<<std::endl;
			ExportWeightBias("min_weight.txt","min_bias.txt");
		}
	}
	std::cout<<"\n\t\tLEARNING PROCESS IS FINISHED"<<std::endl;
	InterimResults<<"\n\t\t****************************************"<<std::endl;
	InterimResults<<"\t\t***** LEARNING PROCESS IS FINISHED *****"<<std::endl;
	InterimResults<<"\t\t****************************************"<<std::endl;
	PrintWeight();
	PrintBias();
	ExportWeightBias("trained_weight.txt","trained_bias.txt");
	_exportFiles(0);
}

//get Method
int NeuralNetwork::n_Input(){
	return n_Inputs;
}
int NeuralNetwork::n_Output(){
	return n_Outputs;
}
Matrix NeuralNetwork::getX(){
	return X;
}
Matrix NeuralNetwork::getY(){
	return Y;
}

// FIRST CONSTRUCTOR
NeuralNetwork::NeuralNetwork(const char* filename, int execute = 0) {
	int i, j;

	strcpy(inputs_fileName, filename);
	char TERM[30];
	// define the input file
	std::ifstream input_file(inputs_fileName, std::ios::in);

	/////For TRAINING//////////
	if (execute==0){
		input_file >> TERM;
		std::cout<<"TERM is: "<<TERM<<std::endl;
		if (strcmp(TERM,"input_trainingfileName:")!=0){
			InterimResults<<"Expected input_trainingfileName: But read "<< TERM<<std::endl;
			return;
		}
		input_file >> input_trainingfileName;

		input_file >> TERM;
		std::cout<<"TERM is: "<<TERM<<std::endl;
		if (strcmp(TERM,"output_trainingfileName:")!=0){
			InterimResults<<"Expected output_trainingfileName: But read "<< TERM<<std::endl;
			return;
		}
		input_file >> output_trainingfileName;

		input_file >> TERM;
		std::cout<<"TERM is: "<<TERM<<std::endl;
		if (strcmp(TERM,"NumOfIter")!=0){
			InterimResults<<"Expected NumOfIter But read "<< TERM<<std::endl;
			return;
		}
		input_file >> NumOfIter;

		input_file >> TERM;
		std::cout<<"TERM is: "<<TERM<<std::endl;
		if (strcmp(TERM,"LearningRate")!=0){
			InterimResults<<"Expected LearningRate But read "<< TERM<<std::endl;
			return;
		}
		input_file >> learningRate;
	}
	/////For EXECUTING//////////
	if (execute==1){
		input_file >> TERM;
		std::cout<<"TERM is: "<<TERM<<std::endl;
		if (strcmp(TERM,"weight_fileName:")!=0){
			InterimResults<<"Expected weight_fileName:: But read "<< TERM<<std::endl;
			return;
		}
		input_file >> weight_fileName;
		input_file >> TERM;
		std::cout<<"TERM is: "<<TERM<<std::endl;
		if (strcmp(TERM,"bias_fileName:")!=0){
			InterimResults<<"Expected bias_fileName: But read "<< TERM<<std::endl;
			return;
		}
		input_file >> bias_fileName;
		input_file >> TERM;
		std::cout<<"TERM is: "<<TERM<<std::endl;
		if (strcmp(TERM,"input_fileName:")!=0){
			InterimResults<<"Expected input_fileName: But read "<< TERM<<std::endl;
			return;
		}
		input_file >> realInput_fileName;
	}
	
	input_file >> TERM;
	std::cout<<"TERM is: "<<TERM<<std::endl;
	if (strcmp(TERM,"NumOfInputs")!=0){
		InterimResults<<"Expected NumOfInputs But read "<< TERM<<std::endl;
		return;
	}
	input_file >> n_Inputs;
	std::cout<<"NumOfInputs is: "<<n_Inputs<<std::endl;

	input_file >> TERM;
	std::cout<<"TERM is: "<<TERM<<std::endl;
	if (strcmp(TERM,"NumOfOutputs")!=0){
		InterimResults<<"Expected NumOfOutputs But read "<< TERM<<std::endl;
		return;
		}
	input_file >> n_Outputs;
	input_file >> TERM;
	std::cout<<"TERM is: "<<TERM<<std::endl;
	if (strcmp(TERM,"NumOfSamples")!=0){
		InterimResults<<"Expected NumOfSamples But read "<< TERM<<std::endl;
		return;
		}
	input_file >> n_Samples;
	input_file >> TERM;
	if (strcmp(TERM,"NumOfHiddenLayers")!=0){
		InterimResults<<"Expected NumOfHiddenLayers But read "<< TERM<<std::endl;
		return;
		}
	input_file >> n_HLayers;
	std::cout << "No of Inputs: \t" << n_Inputs << std::endl;;
	std::cout << "No of Outputs: \t" << n_Outputs << std::endl;
	std::cout << "No of Samples: \t" << n_Samples << std::endl;
	std::cout << "No of HLayers: \t" << n_HLayers << std::endl;

	
	//Vector n_NeuronsPerlayer stores No of neurons per layer (include input layer and output layer)
	n_NeuronsPerLayers.push_back(n_Inputs); // first element is n_Inputs
	int value;

	input_file >> TERM;
	if (strcmp(TERM,"NumOfNeuronsPerHiddenLayer")!=0){
		InterimResults<<"Expected NumOfNeuronsPerHiddenLayer But read "<< TERM<<std::endl;
		return;
	}

	for(i=0; i <n_HLayers; i++){
		//store n_NeuronsPerHiddenLayer to Vectore n_NeuronsPerHLayer
		input_file >> value;
		n_NeuronsPerHLayer.push_back(value);
		//Also push data into n_NeuronsPerLayer vector
		n_NeuronsPerLayers.push_back(n_NeuronsPerHLayer[i]);
		std::cout << "HiddenLayer: " << i+1 << " has " << n_NeuronsPerHLayer[i] <<" neuron(s) " << std::endl;
		//InterimResults <<"HiddenLayer: " << i+1 << " has " << n_NeuronsPerHLayer[i] <<" neuron(s) " << std::endl;
	}
	n_NeuronsPerLayers.push_back(n_Outputs); // last element will be n_Output
	_createDirectories(execute);
	/////////////////////////////////////////////////////////////////////////////////////
	input_file.close();
	input_file.clear();
	//fs::copy (inputs_fileName,directory +"/"+ inputs_fileName);
	
	InterimResults.open("InterimResults.txt", std::ios::out);//|std::ios::app);//directory + "/" + 
	InterimResults << "\t\t*** BUILDING INSTANCE STRUCTURE ***" << std::endl;
	LOG_START;
	/////Print to Process file////////

	InterimResults<< "No of Iteration: \t" << NumOfIter << std::endl;;
	InterimResults<< "Learning Rate: \t" << learningRate << std::endl;;
	InterimResults<< "No of Inputs: \t" << n_Inputs << std::endl;;
	InterimResults<< "No of Outputs: \t" << n_Outputs << std::endl;
	InterimResults<< "No of Samples: \t" << n_Samples << std::endl;
	InterimResults<<"No of HLayers: \t" << n_HLayers << std::endl;
	for(i=0; i <n_HLayers; i++)
	{
		InterimResults <<"HiddenLayer: " << i+1 << " has " << n_NeuronsPerHLayer[i] <<" neuron(s) " << std::endl;
	}
	//InterimResults <<"Working directory:\n"<<directory<<std::endl;
	InterimResults << "\t\t*** INSTANCE STRUCTURE CREATED ***" << std::endl;
	LOG_END;
}

// Take sameple inputs and outputs from text file 
void NeuralNetwork::InputOutputTraining (){//const char * filenameIn, const char * filenameOut){
	
	std::ofstream normalize_output("normalize.txt", std::ios::out);//directory +"/"+
	InterimResults <<std::endl;
	InterimResults << "\t\t*** PROCESSING INPUT DATA ***" << std::endl;
	LOG_START;
	int i,j;

	// define the input file for Input Matrix and Output Matrix
	//strcpy(inputs_fileName, filenameIn);
	std::ifstream in_input_file(input_trainingfileName, std::ios::in);
	//strcpy(outputs_fileName,filenameOut);
	std::ifstream in_output_file(output_trainingfileName, std::ios::in);

	float value;//temp input value
	//std::cout<<value<<std::endl;
			///////////////////MATRIX X/////////////////////////
	// Matrix X (n_Samples,n_Input)
	//Create matrix X from input file (Training input samples)
	X.setRow(n_Samples);
	X.setCol(n_Inputs);
	std::vector<std::vector<float>> tempIn(n_Samples);
	for (i = 0; i < n_Samples; i++) {
		for (j = 0; j < n_Inputs; j++) {
			in_input_file >> value;
			tempIn[i].push_back(value);
			//std::cout << "X[" << i << "][" << j <<"] " << temp[i][j] << "\t";
		}
	}
	X.setArray(tempIn); // copy temp matrix into input matrix X
	std::cout<<"Maxtrix X: \n"<<X.getArray();//X.getArray();
	InterimResults<<"Maxtrix X: \n"<<X.getArray();
	
	tempIn.clear(); 	  // clear tempIn matrix
	in_input_file.close();
	in_input_file.clear();
	
	//InterimResults<<"X mean: "<<X.mean()<<std::endl;
	std::cout<<"X normalized: "<<X.normalizeMinMax()<<std::endl;
	//InterimResults<<"X normalized: "<<X.normalizeMinMax()<<std::endl;
	
	///////////////EXPORT X MIN AND MAX VALUE for EXCECUTE PART///////////////
	normalize_output<<"Input:\n";
	normalize_output<<"MinVector:\n";
	for(int i=0; i<n_Inputs; ++i){
  		normalize_output <<X.getVectorMin()[i] << '\t';
	}
	normalize_output <<"\nMaxVector:\n";
	for(int i=0; i<n_Inputs; ++i){
  		normalize_output <<X.getVectorMax()[i] << '\t';
	}
	normalize_output<<std::endl;
	///////////////////////////////////////////////////////////////////////////

	
	///////////////////MATRIX Y/////////////////////////
	// Matrix Y (n_Samples,n_Outputs)
	//Create matrix Y from sample output file (expected output value)
	Y.setRow(n_Samples);
	Y.setCol(n_Outputs);
	std::vector<std::vector<float>> tempOut(n_Samples);
	for(i=0;i<n_Samples;i++){
		for(j=0;j<n_Outputs;j++){
			in_output_file>>value;
			tempOut[i].push_back(value);
		}
	}
	Y.setArray(tempOut); // copy tempOut matrix into output matrix Y
	std::cout<<"Maxtrix Y: \n"<<Y.getArray();
	InterimResults<<"Maxtrix Y: \n"<<Y.getArray();
	tempOut.clear(); 	  // clear temp matrix
	in_output_file.close();
	in_output_file.clear();
	/////////////////////////////////////////////////
	std::cout<<Y.normalizeMinMax()<<std::endl;
	///////////////EXPORT Y MIN AND MAX VALUE for EXCECUTE PART///////////////
	normalize_output<<"Output:\n";
	normalize_output<<"MinVector:\n";
	for(int i=0; i<n_Outputs; ++i){//Y.getVectorMean().size()
  		normalize_output <<Y.getVectorMin()[i] << '\t';
	}
	normalize_output<<"\nMaxVector:\n";
	for(int i=0; i<n_Outputs; ++i){
  		normalize_output <<Y.getVectorMax()[i] << '\t';
	}
	normalize_output<<std::endl;

	
	///////////////////////////////////////////////////////////////////////////
	//fs::copy_file("normalize.txt",directory +"/"+"normalize.txt",fs::copy_options::overwrite_existing);
	InterimResults << "*** SAMPLE INPUT AND OUTPUT VALUES POPULATED ***" << std::endl;
	LOG_END;
}


//Take Weights and Bias from input files
void NeuralNetwork::SetWeightsBias(){
	InterimResults <<std::endl;
	InterimResults << "\t\t*** WEIGHT AND BIAS INSERTION ***" << std::endl;
	LOG_START;
	int i,j,k;
	std::string temp;

	std::ifstream input_weights(weight_fileName, std::ios::in);
	std::ifstream input_bias(bias_fileName, std::ios::in);

	float value;//temp input value
		///////////////////MATRIX W/////////////////////////
	//Take weights and bias from input file and create 3D Array Matrix Weight and Bias
	for (i=0;i<=n_HLayers;i++){
		for (j = 0; j < n_NeuronsPerLayers[i]; j++) {
			for (k = 0; k < n_NeuronsPerLayers[i+1]; k++) {
				input_weights >> temp;
				//InterimResults<<temp<<std::endl;
				if (temp == "Weights["+std::to_string(i)+"]["+std::to_string(j)+"]["+std::to_string(k)+"]="){
					input_weights >> value;
					Weights[i][j][k]=value;
				}
				else{
					InterimResults<<"Input Weight Format doesn't match with Struct File"<<std::endl;
					return;
				}
			}
		}
	}
	PrintWeight();
	InterimResults << "\t\t*** WEIGHTS ARE INSERTED ***" << std::endl;
	input_weights.close();
	input_weights.clear();
	
		///////////////////MATRIX B/////////////////////////
	for (j=0;j<=n_HLayers;j++){
		//std::cout<<"\tBias at layer: "<<j<<std::endl;
		for (k = 0; k < n_NeuronsPerLayers[j+1]; k++) {
			input_bias >> temp;
			//InterimResults<<temp<<std::endl;
			if (temp == "Bias["+std::to_string(j)+"][0]["+std::to_string(k)+"]="){
				input_bias >> value;
				Bias[j][0][k] = value;
			}
			else{
					InterimResults<<"Input Bias Format doesn't match with Struct File"<<std::endl;
					return;
				}
		}
	}
	std::cout << std::endl;
	PrintBias();
	InterimResults << "\t\t*** BIAS ARE INSERTED ***" << std::endl;
	input_bias.close();
	input_bias.clear();
	LOG_END;
}
void NeuralNetwork::SetWeightsBias(const char * weights_file, const char * bias_file){
	InterimResults <<std::endl;
	InterimResults << "\t\t*** WEIGHT AND BIAS INSERTION ***" << std::endl;
	LOG_START;
	int i,j,k;
	std::string temp;

	strcpy(inputs_fileName, weights_file);
	std::ifstream input_weights(inputs_fileName, std::ios::in);
	strcpy(outputs_fileName,bias_file);
	std::ifstream input_bias(outputs_fileName, std::ios::in);

	float value;//temp input value
		///////////////////MATRIX W/////////////////////////
	//Take weights and bias from input file and create 3D Array Matrix Weight and Bias
	for (i=0;i<=n_HLayers;i++){
		for (j = 0; j < n_NeuronsPerLayers[i]; j++) {
			for (k = 0; k < n_NeuronsPerLayers[i+1]; k++) {
				input_weights >> temp;
				//InterimResults<<temp<<std::endl;
				if (temp == "Weights["+std::to_string(i)+"]["+std::to_string(j)+"]["+std::to_string(k)+"]="){
					input_weights >> value;
					Weights[i][j][k]=value;
				}
				else{
					InterimResults<<"Input Weight Format doesn't match with Struct File"<<std::endl;
					return;
				}
			}
		}
	}
	PrintWeight();
	InterimResults << "\t\t*** WEIGHTS ARE INSERTED ***" << std::endl;
	input_weights.close();
	input_weights.clear();

		///////////////////MATRIX B/////////////////////////
	for (j=0;j<=n_HLayers;j++){
		//std::cout<<"\tBias at layer: "<<j<<std::endl;
		for (k = 0; k < n_NeuronsPerLayers[j+1]; k++) {
			input_bias >> temp;
			//InterimResults<<temp<<std::endl;
			if (temp == "Bias["+std::to_string(j)+"][0]["+std::to_string(k)+"]="){
				input_bias >> value;
				Bias[j][0][k] = value;
			}
			else{
					InterimResults<<"Input Bias Format doesn't match with Struct File"<<std::endl;
					return;
				}
		}
	}
	std::cout << std::endl;
	PrintBias();
	InterimResults << "\t\t*** BIAS ARE INSERTED ***" << std::endl;
	input_bias.close();
	input_bias.clear();
	LOG_END;
}

void NeuralNetwork::Normalized(){
	H=X.normalizeMinMax(); //Set H = X	 /////////////////////////////NEED TO PUT IN TRAINING CLASS
}
void NeuralNetwork::Normalized(std::vector<float> v1,std::vector<float> v2){
	H=X.normalizeMinMax(v1,v2);
	std::cout<<H.getArray()<<std::endl;
}

void NeuralNetwork::Denormalized(){
	H=H.deNormalizeMinMax(Y.getVectorMin(),Y.getVectorMax());
}
void NeuralNetwork::Denormalized(std::vector<float> v1,std::vector<float> v2){
	H=H.deNormalizeMinMax(v1,v2);
	for(int i=0;i<n_Samples;i++){
		for(int j=0;j<n_Outputs;j++){
		}
	}
	std::cout<<H.getArray()<<std::endl;
	InterimResults<<"Output value(s) after Denormalization:\n"<<H.getArray()<<std::endl;
}
void NeuralNetwork::SetInput(){//const char * input_file){
	InterimResults <<std::endl;
	InterimResults << "\t\t*** INPUT VALUES INSERTION ***" << std::endl;
	LOG_START;
	int i = 0,j;
	//strcpy(inputs_fileName, input_file);
	std::ifstream data(realInput_fileName, std::ios::in);
	float value;
	std::vector <std::vector <float> > temp;
	//while (!data.eof()){
	for (i=0;!data.eof();i++){
		temp.push_back(std::vector <float> ());
		for(j=0;j<2;j++){
			//std::cout<<"hello"<<std::endl;
			data>>value;
			//std::cout<<value<<' ';
			temp[i].push_back(value);
			std::cout<<"X["<<i<<"]["<<j<<"]= "<<temp[i][j]<<' ';
		}
		std::cout<<std::endl;
		//i++;
	}
	//std::cout<<temp.size()<<" "<<temp[0].size();
	X.setArray(temp);
	X.setRow(n_Samples);//temp.size());
	X.setCol(n_Inputs);//temp[0].size());
	InterimResults<<"Input:\n"<<X.getArray()<<std::endl;
	temp.clear();
	data.close();
	data.clear();
	LOG_END;
}


//Randomize Weights and Bias
//Without declaring innitial Weights and Bias, they will be randomized
void NeuralNetwork::RandomizeWeights(){
	InterimResults <<std::endl;
	InterimResults << "\t\t*** INITIALIZING WEIGHT AND BIAS ***" << std::endl;
	LOG_START;
	int i,j,k;
	/////////////////////// Randomize Weights /////////////////////////////////
	// Randomize Weight based on n_NeuronsPerHiddenLayer and n_NeuronsPerLayer
	for (i=0;i<=n_HLayers;i++){
		std::cout<<"\tWeight at layer: "<<i<<std::endl;
		for (j = 0; j < n_NeuronsPerLayers[i]; j++) {
			for (k = 0; k < n_NeuronsPerLayers[i+1]; k++) {
				//Weights[i][j][k] = ((std::rand()/float(RAND_MAX) *0.2)-0.1); // 2)/10);
				Weights[i][j][k] = ((std::rand()/float(RAND_MAX) *2)/10); 
				//std::cout << " Weights["<<i<<"][" << j << "][" << k<< "] = " << Weights[i][j][k] << "\t";
				//InterimResults <<" Weights["<<i<<"][" << j << "][" << k<< "] = " << Weights[i][j][k] << "\t";
			}
			//std::cout << std::endl;
			//InterimResults <<std::endl;
		}
	}
	PrintWeight();
	std::cout << std::endl;
	InterimResults << "\t\t*** WEIGHTS RANDOMIZED ***" << std::endl;

	/////////////////////// Randomize Bias /////////////////////////////////
	for (j=0;j<=n_HLayers;j++){
		std::cout<<"\tBias at layer: "<<j<<std::endl;
		for (k = 0; k < n_NeuronsPerLayers[j+1]; k++) {
			//std::srand(std::time(NULL)+i+j+k);
			//std::srand(std::time(0)-i-j-k);
			Bias[j][0][k] =0;//((std::rand()/float(RAND_MAX) * 8) - 4);
			//std::cout << " Bias["<<j<<"][0][" << k<< "] = " << Bias[j][0][k] << "\t";
			//InterimResults <<" Bias["<<j<<"][0][" << k<< "] = " << Bias[j][0][k] << "\t";
		}
		//std::cout << std::endl;
		//InterimResults <<std::endl;
	}
	PrintBias();
	InterimResults << "\t\t*** BIAS INITIALIZED ***" << std::endl;
	std::cout << std::endl;
	InterimResults << "\t\t*** BIAS ARE CREATED and POPULATED ***" << std::endl;
	LOG_END;
}

// FORWARD PROPAGATION
void NeuralNetwork::ForwardPropagation(){
	InterimResults <<std::endl;
	InterimResults << "\t\t*** FORWARD PROPAGATION ***" << std::endl;
	LOG_START;
	int i,j,k;
		H=X.normalizeMinMax(); //Set H = X	
	for(i=0;i<n_NeuronsPerLayers.size()-1;i++){
		//H=H.applyFunction(ReLU);
		std::cout<<H.getArray();
		std::vector<std::vector<float>> temp(n_NeuronsPerLayers[i]); 
		/////////////////////////////LOAD value to Matrix W///////////////////////////////////////////
		for(j=0;j<n_NeuronsPerLayers[i];j++){
			for(k=0;k<n_NeuronsPerLayers[i+1];k++){
				temp[j].push_back(Weights[i][j][k]);
			}
		}
		// Row and Col of matrix W depends on #of neurons in this layer and the next one
		W.setRow(n_NeuronsPerLayers[i]); 
		W.setCol(n_NeuronsPerLayers[i+1]);
		W.setArray(temp);
		temp.clear();
		std::cout<<"Matrix W at layer "<<i<<" :\n"<<W.getArray()<<std::endl;
		InterimResults<<"Matrix W at layer "<<i<<" :\n"<<W.getArray()<<std::endl;


		//////////////////LOAD value from Bias to Matrix B///////////////////////////////////////////////////
		std::vector<std::vector<float>> temp1(H.getRow());
		for (j=0;j<H.getRow();j++){
			for (k = 0; k <W.getCol(); k++) {
				temp1[j].push_back(Bias[i][0][k]); //Bias only has ROW = 1
			}
		}
		// Row and Col of matrix B depends on Row of matrix H and Col of matrix W respectively
		// Because size of matrix B has to be the same with size of H.W to be able to do addition
		B.setRow(H.getRow());
		B.setCol(W.getCol());
		B.setArray(temp1);
		temp1.clear(); 	 // clear temp matrix
		//std::cout<<"Matrix B:2 \n"<<B.getArray()<<std::endl;
		std::cout<<"Matrix B at layer "<<i<<" :\n"<<B.getArray()<<std::endl;
		InterimResults<<"Matrix B at layer "<<i<<" :\n"<<B.getArray()<<std::endl;

		//////////////////////////MATRIX H//////////////////////////////////////////////////////
		//update Matrix H as H.W+B
		H = H.dot(W).add(B);
		//std::cout<<"Matrix H:2 \n"<<H.getArray()<<std::endl;
		std::cout<<"Matrix H at layer "<<i<<" :\n"<<H.getArray();
		InterimResults<<"Matrix H at layer "<<i<<" :\n"<<H.getArray()<<std::endl;
		//store H into 3D array Z
		for(j=0;j<H.getRow();j++){
			for(k=0;k<H.getCol();k++){
				Z[i][j][k] = H.getArray()[j][k];
				std::cout<<"Z["<<i<<"]["<<j<<"]["<<k<<"]= "<<Z[i][j][k]<<"\t";
			}
			std::cout<<std::endl;
		}
		//H=H.applyFunction(ReLU); 
		
		if(i != n_NeuronsPerLayers.size()-2){ //CHECK -- will not apply RELU to OUTPUT layer
			H=H.applyFunction(ReLU); 
		}
	}
	LOG_END;
}

//COST FUNCTION
void NeuralNetwork::CostFunction(){
	InterimResults <<std::endl;
	InterimResults << "\t\t*** COST OF THE FUNCTION ***" << std::endl;
	LOG_START;
	
	//Cost function J = 1/2 * sqrt ( (real output - expect output)^2)
	J = ((H.subtract(Y.normalizeMinMax())).sumElementSquare())/2; //applyFunction(ReLU)
	
	output_excelfile<<"\t"<<J<<std::endl;

	std::cout << "Cost of the Function = "<<J<<std::endl;
	InterimResults << "Cost of the Function = "<<J<<std::endl;
	InterimResults << "*** COST OF FUNCTION IS CALCULATED ***" << std::endl;
	cost_output << J << std::endl;
	LOG_END;
}

// BACK PROPAGATION
void NeuralNetwork::BackPropagation(float learningRate){
	InterimResults <<std::endl;
	InterimResults << "\t\t*** BACK PROPAGATION ***" << std::endl;
	InterimResults << "\t\t*** WEIGHT AND BIAS UPDATING ***" << std::endl;
	LOG_START;
	std::cout << "\t\t*** BACK PROPAGATION ***" << std::endl;
	int i,j,k;

	float learnRate=learningRate;//usually 0.05--learning RATE

	std::cout<<H.getArray();
	std::cout<<H.applyFunction(ReLU).getArray();
	std::cout<<H.applyFunction(ReLUprime).getArray();
	// dJ/dB
	dB = (H.applyFunction(ReLU).subtract(Y.normalizeMinMax()).multiply(H.applyFunction(ReLUprime)));

	////////////////////////////////////UPDATE LAST LAYER BIAS////////////////////////////////////////////////////////////////
		int lastlayer= n_NeuronsPerLayers.size()-2;
		std::cout<<"\tUpdate Bias at layer: "<<lastlayer<<std::endl;
		InterimResults << "\tUpdate Bias at layer: "<<lastlayer<<std::endl;
		for (k = 0; k < n_NeuronsPerLayers[lastlayer+1]; k++) {
			Bias[lastlayer][0][k] = Bias[lastlayer][0][k]- learnRate*dB.mean().getArray()[0][k];      //B=B-rate*dB
			std::cout << " Bias["<<lastlayer<<"][0][" << k<< "] = " << Bias[lastlayer][0][k] << "\t";
			InterimResults << " Bias["<<lastlayer<<"][0][" << k<< "] = " << Bias[lastlayer][0][k] << "\t";
		}
		std::cout << std::endl;
		InterimResults << std::endl;
	//////////////////////////////////////////////////////

	for(i=n_NeuronsPerLayers.size()-2;i>0;i--){
		std::vector<std::vector <float> > temp(n_NeuronsPerLayers[i]); 

		///////////////////////////////LOAD value from Bias to Matrix B///////////////////////////////////////////////////
		std::vector<std::vector<float>> temp1(H.getRow());
		for (j=0;j<H.getRow();j++){ 
			for (k = 0; k <W.getCol(); k++) {
				temp1[j].push_back(Bias[i][0][k]); 			
			}
		}
		B.setRow(H.getRow()); 
		B.setCol(W.getCol());
		B.setArray(temp1);
		temp1.clear(); 	 // clear temp matrix
		
		std::cout<<"Matrix B at layer "<<i<<" :\n"<<B.getArray()<<std::endl;


		///////////////////////////LOAD value from Weight to Matrix W/////////////////////////////////////////////
		for(j=0;j<n_NeuronsPerLayers[i];j++){
			for(k=0;k<n_NeuronsPerLayers[i+1];k++){
				temp[j].push_back(Weights[i][j][k]);
			}
		}
		W.setRow(n_NeuronsPerLayers[i]); 
		W.setCol(n_NeuronsPerLayers[i+1]); 
		W.setArray(temp);
		temp.clear();
		std::cout<<"Matrix W at layer "<<i<<" :\n"<<W.getArray()<<std::endl;

		///////////////////////////////////////////////////////////////////////////////////////////////////////////
	
		H.setRow(B.getRow()); 
		H.setCol(W.getRow()); 
	
		std::vector<std::vector<float>> temp2(H.getRow());
		for (j=0;j<H.getRow();j++){ 
			for (k = 0; k <H.getCol(); k++) {
				temp2[j].push_back(Z[i-1][j][k]); 
			}
		}
		H.setArray(temp2);
		temp2.clear(); 	 // clear temp matrix
		std::cout<<"Matrix H at layer "<<i-1<<" :\n"<<H.getArray()<<std::endl;
		
		std::cout<<H.applyFunction(ReLU);
		std::cout<<H.applyFunction(ReLUprime);

		//////////////////////////////////GRADIENT///////////////////////////////////////////////////
		dW=H.applyFunction(ReLU).transpose().dot(dB); // dJ/dW
		dB= dB.dot(W.transpose()).multiply(H.applyFunction(ReLUprime)); // dJ/dB
		/////////////////////////////////////////////////////////////////////////////////////////////


		////////////////////////////////Update Weights//////////////////////////////////////////
		std::cout<<"\tUpdate Weight at layer: "<<i<<std::endl;
		InterimResults << "\tUpdate Weight at layer: "<<i<<std::endl;
		for (j = 0; j < n_NeuronsPerLayers[i]; j++) {
			for (k = 0; k < n_NeuronsPerLayers[i+1]; k++) {
				Weights[i][j][k] = Weights[i][j][k]-(learnRate*dW.getArray()[j][k]);
				std::cout << " Weights["<<i<<"][" << j << "][" << k<< "] = " << Weights[i][j][k] << "\t";
				InterimResults << " Weights["<<i<<"][" << j << "][" << k<< "] = " << Weights[i][j][k] << "\t";
				
			}
			std::cout << std::endl;
			InterimResults << std::endl;
		}
		////////////////////////////////UPDATE BIAS//////////////////////////////////////////
		std::cout<<"\tUpdate Bias at layer: "<<i-1<<std::endl;
		InterimResults <<"\tUpdate Bias at layer: "<<i-1<<std::endl;
		for (k = 0; k < n_NeuronsPerLayers[i]; k++) {
			Bias[i-1][0][k] = Bias[i-1][0][k]- learnRate*dB.mean().getArray()[0][k];         //////////////////UPDATE BIAS
			std::cout << " Bias["<<i-1<<"][0][" << k<< "] = " << Bias[i-1][0][k] << "\t";
			InterimResults <<" Bias["<<i-1<<"][0][" << k<< "] = " << Bias[i-1][0][k] << "\t";
		}
		std::cout<<std::endl;
		InterimResults<<std::endl;
	}
	
	//////////////////////////////////////////////////////
	dW= X.transpose().dot(dB);
	/////////////////////////////////////////////////////
	///////////////////UPDATE FIRST LAYER WEIGHT//////////////////////////////
	std::cout<<"\tUpdate Weight at layer: "<<i<<std::endl;
	InterimResults<<"\tUpdate Weight at layer: "<<i<<std::endl;
		for (j = 0; j < n_NeuronsPerLayers[i]; j++) {
			for (k = 0; k < n_NeuronsPerLayers[i+1]; k++) {
				Weights[i][j][k] = Weights[i][j][k]-(learnRate*dW.getArray()[j][k]);
				std::cout << " Weights["<<i<<"][" << j << "][" << k<< "] = " << Weights[i][j][k] << "\t";
				InterimResults<< " Weights["<<i<<"][" << j << "][" << k<< "] = " << Weights[i][j][k] << "\t";
			}
			std::cout << std::endl;
			InterimResults << std::endl;
		}
	InterimResults << "\t\t*** WEIGHTS AND BIAS ARE UPDATED ***" << std::endl;	
	std::cout<<std::endl;
	LOG_END;
}


void NeuralNetwork::ExportWeightBias(const char* weight_file ="trained_weight.txt",const char* bias_file ="trained_bias.txt"){
	std::ofstream trained_weight_output(weight_file, std::ios::out);//directory +"/"+ 
	std::ofstream trained_bias_output(bias_file, std::ios::out);//directory +"/"+ 
	//LOG_START;
	int i,j,k;
	//Print out New Weight and Bias that stored in 3D array Weight and Bias
	for (k=0;k<=n_HLayers;k++){
		std::cout<<"\tWeight at layer: "<<k<<std::endl;
		//InterimResults<<"\tWeight at layer: "<<k<<std::endl;
		for (i = 0; i < n_NeuronsPerLayers[k]; i++) {
			for (j = 0; j < n_NeuronsPerLayers[k+1]; j++) {
				std::cout << "Weights["<<k<<"][" << i << "][" << j<< "]= " << Weights[k][i][j] << "\t";
				//InterimResults << " Weights["<<k<<"][" << i << "][" << j<< "] = " << Weights[k][i][j] << "\t";
				trained_weight_output<< "Weights["<<k<<"][" << i << "][" << j<< "]= " << Weights[k][i][j] << "\t";
			}
			std::cout << std::endl;
			//InterimResults << std::endl;
			trained_weight_output<<std::endl;
		}
	}
	std::cout<<std::endl;
	
	for (k=0;k<=n_HLayers;k++){
		std::cout<<"\tBias at layer: "<<k<<std::endl;
		//InterimResults<<"\tBias at layer: "<<k<<std::endl;
		for (j = 0; j < n_NeuronsPerLayers[k+1]; j++) {
			std::cout << "Bias["<<k<<"][0][" << j<< "]= " << Bias[k][0][j] << "\t";
			//InterimResults << " Bias["<<k<<"][0][" << j<< "] = " << Bias[k][0][j] << "\t";
			trained_bias_output<< "Bias["<<k<<"][0][" << j<< "]= " << Bias[k][0][j] << "\t";
		}
		std::cout << std::endl;
		//InterimResults << std::endl;
		trained_bias_output<< std::endl;
	}

	//fs::copy (weight_file,directory+"/"+weight_file,fs::copy_options::overwrite_existing);
	//fs::copy (bias_file,directory+"/"+bias_file,fs::copy_options::overwrite_existing);
	//LOG_END;
}

void NeuralNetwork::ExpectedOutPut(float sleep, float study){
	InterimResults <<std::endl;
	InterimResults << "\t\t*** EXPECTED SOLUTION ***" << std::endl;
	LOG_START;
	int i,j,k;
	std::vector<std::vector<float>> input(1);

	input[0].push_back(sleep);
	input[0].push_back(study);
	
	H.setArray(input);
	//H=H.normalize(X.getVectorMean(),X.getVectorMax());
	//std::cout<<temp.getArray();
	//H=temp;
	H.setCol(2);
	H.setRow(1);
	H=H.normalizeMinMax(X.getVectorMin(),X.getVectorMax());
	std::cout<<"H after normalize: \n"<<H.getArray();
	//temp.setCol(2);
	//temp.setRow(1);
	//temp.setArray(input);
	//Set H = X	
	//InterimResults<<"Matrix H "<<temp.getArray()<<std::endl;
	
	for(i=0;i<n_NeuronsPerLayers.size()-1;i++){
		H=H.applyFunction(ReLU);
		std::cout<<"H after applied RELU: \n"<<H.getArray();
		std::vector<std::vector<float>> temp(n_NeuronsPerLayers[i]); 

		/////////////////////////////LOAD value to Matrix W///////////////////////////////////////////
		for(j=0;j<n_NeuronsPerLayers[i];j++){
			for(k=0;k<n_NeuronsPerLayers[i+1];k++){
				temp[j].push_back(Weights[i][j][k]);
			}
		}
		// Row and Col of matrix W depends on #of neurons in this layer and the next one
		W.setRow(n_NeuronsPerLayers[i]); 
		W.setCol(n_NeuronsPerLayers[i+1]);
		W.setArray(temp);
		temp.clear();
		std::cout<<"Matrix W at layer "<<i<<" :\n"<<W.getArray()<<std::endl;
		InterimResults<<"Matrix W at layer "<<i<<" :\n"<<W.getArray()<<std::endl;


		//////////////////LOAD value from Bias to Matrix B///////////////////////////////////////////////////
		std::vector<std::vector<float>> temp1(H.getRow());
		for (j=0;j<H.getRow();j++){
			for (k = 0; k <W.getCol(); k++) {
				temp1[j].push_back(Bias[i][0][k]); //Bias only has ROW = 1
			}
		}
		// Row and Col of matrix B depends on Row of matrix H and Col of matrix W respectively
		// Because size of matrix B has to be the same with size of H.W to be able to do addition
		B.setRow(H.getRow());
		B.setCol(W.getCol());
		B.setArray(temp1);
		temp1.clear(); 	 // clear temp matrix
		//std::cout<<"Matrix B:2 \n"<<B.getArray()<<std::endl;
		std::cout<<"Matrix B at layer "<<i<<" :\n"<<B.getArray()<<std::endl;
		InterimResults<<"Matrix B at layer "<<i<<" :\n"<<B.getArray()<<std::endl;

		//////////////////////////MATRIX H//////////////////////////////////////////////////////
		//update Matrix H as H.W+B
		H=H.dot(W).add(B);
		//std::cout<<"Matrix H:2 \n"<<H.getArray()<<std::endl;
		std::cout<<"Matrix H at layer "<<i<<" :\n"<<H.getArray();
		InterimResults<<"Matrix H at layer "<<i<<" :\n"<<H.getArray();		
	}
	H=H.deNormalizeMinMax(Y.getVectorMin(),Y.getVectorMax());
	std::cout<<"Matrix H after denormalized:\n"<<H.getArray();
	LOG_END;
}

void NeuralNetwork::PrintWeight(){
	int i,j,k;
	for (i=0;i<=n_HLayers;i++){
		for (j = 0; j < n_NeuronsPerLayers[i]; j++) {
			for (k = 0; k < n_NeuronsPerLayers[i+1]; k++) {
				std::cout << "Weights["<<i<<"][" << j << "][" << k<< "]= " << Weights[i][j][k] << "\t";
				InterimResults <<"Weights["<<i<<"][" << j << "][" << k<< "]= " << Weights[i][j][k] << "\t";
			}
			std::cout << std::endl;
			InterimResults <<std::endl;
		}
	}
}
void NeuralNetwork::PrintBias(){
	int i,j,k;
	for (j=0;j<=n_HLayers;j++){
		//std::cout<<"\tBias at layer: "<<j<<std::endl;
		for (k = 0; k < n_NeuronsPerLayers[j+1]; k++) {
			std::cout << "Bias["<<j<<"][0][" << k<< "]= " << Bias[j][0][k] << "\t";
			InterimResults <<"Bias["<<j<<"][0][" << k<< "]= " << Bias[j][0][k] << "\t";
		}
		std::cout << std::endl;
		InterimResults <<std::endl;
	}
}

void NeuralNetwork::_createDirectories(int execute){
	std::string str;
	if (execute == 0){
		str="case/SleepStudy/Train/";
	}else {str="case/SleepStudy/Execute/";}
    //std::cout<<str<<std::endl;
    str= str + "I" + std::to_string(n_Inputs);
    //std::cout<<str<<std::endl;
    str= str + "H" + std::to_string(n_HLayers);
    //std::cout<<str<<std::endl;
    for (int i=0;i<n_HLayers;i++){
        str= str + "N" + std::to_string(n_NeuronsPerHLayer[i]);
    }
    //std::cout<<str<<std::endl;
    str= str + "O" +std::to_string(n_Outputs);
    fs::path path = fs::current_path().parent_path().append(str);

    fs::create_directories(path);
	directory = path;
}
void NeuralNetwork::_exportFiles(int execute){
	if(execute==0){
		fs::copy_file("struct_training.txt",directory +"/"+"struct_training.txt",fs::copy_options::overwrite_existing);
		fs::copy_file("out.xls",directory +"/"+"out.xls",fs::copy_options::overwrite_existing);
		fs::copy_file("min_J.txt",directory +"/"+"min_J.txt",fs::copy_options::overwrite_existing);
		fs::copy_file("min_weight.txt",directory +"/"+"min_weight.txt",fs::copy_options::overwrite_existing);
		fs::copy_file("min_bias.txt",directory +"/"+"min_bias.txt",fs::copy_options::overwrite_existing);
		fs::copy_file("error.txt",directory +"/"+"error.txt",fs::copy_options::overwrite_existing);
	}
	if(execute==1){
		fs::copy_file("struct_executing.txt",directory +"/"+"struct_executing.txt",fs::copy_options::overwrite_existing);
	}
	fs::copy_file("InterimResults.txt",directory +"/"+"InterimResults.txt",fs::copy_options::overwrite_existing);
	fs::copy_file("trained_weight.txt",directory +"/"+"trained_weight.txt",fs::copy_options::overwrite_existing);
	fs::copy_file("trained_bias.txt",directory +"/"+"trained_bias.txt",fs::copy_options::overwrite_existing);
	fs::copy_file("normalize.txt",directory +"/"+"normalize.txt",fs::copy_options::overwrite_existing);
}
