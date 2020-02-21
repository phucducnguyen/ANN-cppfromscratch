#include "Execute.h"
#include <random>
//#include <>

int main(int argc, char* argv[])//int argc, char* argv[]
{	
	float a = 5;
	float b = 10;
	//Execute A("in_nn_struct.txt","trained_weight.txt","trained_bias.txt","normalize.txt");
	Execute A("struct_executing.txt","normalize.txt");//"ANN/case/SleepStudy/Train/I2H2N3N3O1/min_weight.txt","ANN/case/SleepStudy/Train/I2H2N3N3O1/min_bias.txt","ANN/case/SleepStudy/Train/I2H2N3N3O1/normalize.txt"); "min_weight.txt","min_bias.txt",
	//A.NormalizeParameter("normalize.txt");
	A.ExpectedOutPut();//"new_Input.txt");
	//mkdir();
	return 0;
}
