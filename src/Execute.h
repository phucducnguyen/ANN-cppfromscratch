#pragma once
#include "NeuralNetwork.h"
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <time.h>

//Declared output file in constructor
//std::ofstream output_file("execute_draft.txt", std::ios::out);
/* 
#define LOG_START output_file << "*** Process Begins ***" << std::endl
#define LOG_END output_file << "*** Process Ends ***" << std::endl
*/
class Execute {
    public: //public interface for this class
    Execute (const char * struct_file, const char * normalize_file); //Constructor  const char * weights_file, const char * bias_file,
		//void InputsToOutputs();
		//void NormalizeParameter(const char * normalize_file);
		//void ExpectedOutPut(float sleep, float study);
		void ExpectedOutPut();//const char * input_file);
		void _setNormalizedParameter(const char * normalize_file);
    private:
		NeuralNetwork NN;
		std::vector<float> vectorMeanX;
		std::vector<float> vectorMaxX;
		std::vector<float> vectorMinX;
		std::vector<float> vectorMeanY;
		std::vector<float> vectorMaxY;
		std::vector<float> vectorMinY;
		char inputs_fileName[50]; // hold the inputs file name
		char outputs_fileName[50]; // hold the outputs file name
};


Execute::Execute(const char * structfile, 
								const char * normalize_file) {//const char * weights_file, const char * bias_file,
	//NeuralNetwork temp(filename);//NN=temp;
	NN = NeuralNetwork(structfile,1);
	NN.SetWeightsBias();//weights_file,bias_file);
	_setNormalizedParameter(normalize_file);
}
void Execute::ExpectedOutPut(){//const char * input_file){
	NN.SetInput();//input_file);
	NN.Normalized(vectorMinX,vectorMaxX);
	NN.ForwardPropagation();
	NN.Denormalized(vectorMinY,vectorMaxY);
	NN._exportFiles(1);
}

void Execute::_setNormalizedParameter(const char * normalize_file){
	int i,j,k;
	strcpy(inputs_fileName, normalize_file);
	std::ifstream normalize_input(inputs_fileName, std::ios::in);
	float value;
	char check[30];
	normalize_input >> check;
	//std::cout<<check<<std::endl;
	if (strcmp(check,"Input:")!=0){
		InterimResults<<"Expected Input: But read "<<check<<std::endl;
		return;
	}
	normalize_input >> check;
	if (strcmp(check,"MinVector:")!=0){
		InterimResults<<"Expected MinVector: But read "<<check<<std::endl;
		return;
	}
	for (i=0; i<NN.n_Input(); i++){
		normalize_input >>value;
		//std::cout<<value<<' ';
		vectorMinX.push_back(value);
	}

	normalize_input >> check;
	if (strcmp(check,"MaxVector:")!=0){
		InterimResults<<"Expected MaxVector: But read "<<check<<std::endl;
		return;
	}
	//NN.getX().setVectorMean(vectorMean);
	for (i=0; i<NN.n_Input(); i++){
		normalize_input >>value;
		vectorMaxX.push_back(value);
	}
	//NN.getX().setVectorMax(vectorMax);
	normalize_input >> check;
	//std::cout<<check<<std::endl;
	if (strcmp(check,"Output:")!=0){
		InterimResults<<"Expected Output: But read "<<check<<std::endl;
		return;
	}
	normalize_input >> check;
	//std::cout<<check<<std::endl;
	if (strcmp(check,"MinVector:")!=0){
		InterimResults<<"Expected MinVector: But read "<<check<<std::endl;
		return;
	}
	for (i=0; i<NN.n_Output(); i++){
		normalize_input >>value;
		vectorMinY.push_back(value);
	}
	normalize_input >> check;
	//std::cout<<check<<std::endl;
	if (strcmp(check,"MaxVector:")!=0){
		InterimResults<<"Expected MaxVector: But read "<<check<<std::endl;
		return;
	}
	//NN.getY().setVectorMean(vectorMean);
	for (i=0; i<NN.n_Output(); i++){
		normalize_input >>value;
		vectorMaxY.push_back(value);
	}
	//NN.getY().setVectorMax(vectorMax);
	////////////////Print to deBug///////////////////////
	std::cout<<"VectorMeanX: ";
	for (i=0;i<vectorMinX.size();i++){
		std::cout<<vectorMinX[i]<<' ';	
	}
	std::cout<<std::endl;

	std::cout<<"VectorMaxX: ";
	for (i=0;i<vectorMaxX.size();i++){
		std::cout<<vectorMaxX[i]<<' ';
	}
	std::cout<<std::endl;

	std::cout<<"VectorMeanY: ";
	for (i=0;i<vectorMinY.size();i++){
		std::cout<<vectorMinY[i]<<' ';	
	}
	std::cout<<std::endl;

	std::cout<<"VectorMaxY: ";
	for (i=0;i<vectorMaxY.size();i++){
		std::cout<<vectorMaxY[i]<<' ';
	}
	std::cout<<std::endl;
	//////////////////////////////////////////////////////
}
