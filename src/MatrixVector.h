#include <assert.h>
#include <sstream>
#include <vector>
#include <iostream>
#include <cmath>

class Matrix
{
public:
    Matrix();
    Matrix(int row, int col);
    Matrix(std::vector<std::vector<float> > const &array);
    //Matrix(float array[20][50][50]);
    int getRow() const;
    int getCol() const;
    int setRow(int row);
    int setCol(int col);
    
    std::vector<std::vector<float>> getArray();
    std::vector<float> getVectorMean();
    std::vector<float> getVectorMax();
    std::vector<float> getVectorMin();
    std::vector<std::vector<float>> getMinMax();
    float getValue(int row,int col);
    void setArray(std::vector<std::vector<float> > &array);
    void setVectorMean(std::vector<float> meanVector);
    void setVectorMax(std::vector<float> maxVector);
    void setVectorMin(std::vector<float> minVector);


    Matrix equal(Matrix const &m);
    Matrix multiply(float const &value); // scalar multiplication
    Matrix add(Matrix const &m) const; // addition
    Matrix subtract(Matrix const &m) const; // subtraction
    Matrix multiply(Matrix const &m) const; // element by element product (Hadamard product)
    Matrix dot(Matrix const &m) const; // dot product
    Matrix transpose() const; // transposed matrix
    Matrix applyFunction(float (*function)(float)) const; // to apply a function to every element of the matrix
    Matrix mean(); // find mean of the matrix
    ////////////////////Normalization Base on Min and Max value///////////////////////////////
    Matrix normalizeMinMax(); //normalize the matrix
    Matrix normalizeMinMax(std::vector<float>min,std::vector<float>max);
    Matrix deNormalizeMinMax(std::vector<float>min,std::vector<float>max); //de-normalize matrix
    //////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////Normalization Base on Min and Max value///////////////////////////////
    Matrix normalizeMeanMax(); //normalize the matrix
    Matrix normalizeMeanMax(std::vector<float>mean,std::vector<float>max);
    Matrix deNormalizeMeanMax(std::vector<float>mean,std::vector<float>max); //de-normalize matrix
    //////////////////////////////////////////////////////////////////////////////////////////
    float sumElementSquare(); // square every element and sum up
    void print(std::ostream &flux) const;

private:
    std::vector<std::vector<float>> array; // 2D array or 2D matrices
    std::vector<float> vectorMean; //2D array store min and max value for each row
    std::vector<float> vectorMax;
    std::vector<float> vectorMin;
    int row;
    int col;
};
//Overload << operator for print
std::ostream& operator<<(std::ostream &flux, Matrix const &m);
 

//Constructor
Matrix::Matrix(){} 
Matrix::Matrix(int row, int col){
    this->row = row;
    this->col = col;
    this->array = std::vector<std::vector<float>> (row, std::vector<float>(col));
    //
}
 
Matrix::Matrix(std::vector<std::vector<float> > const &array){
    assert(array.size()!=0);
    this->row = array.size();
    this->col = array[0].size();
    this->array = array;
}

//Method
    //set Method
void Matrix::setArray(std::vector<std::vector<float> > &array){
    this->array=array;
}
int Matrix::setRow(int row){
    this->row=row;
}
int Matrix::setCol(int col){
    this->col=col;
}
void Matrix::setVectorMean(std::vector<float> meanVector){
    this->vectorMean = meanVector;
}
void Matrix::setVectorMax(std::vector<float> maxVector){
    this->vectorMax = maxVector;
}
void Matrix::setVectorMin(std::vector<float> minVector){
    this->vectorMin = minVector;
}
    //get Method
std::vector<std::vector<float> > Matrix::getArray(){
    return array;
}
/*std::vector<std::vector<float>> Matrix::getMinMax(){
    return MinMax;
}*/
int Matrix::getRow() const {
    return this->row;
}
int Matrix::getCol() const {
    return this->col;
}
std::vector<float> Matrix::getVectorMax(){
    return vectorMax;
}
std::vector<float> Matrix::getVectorMean(){
    return vectorMean;
}
std::vector<float> Matrix::getVectorMin(){
    return vectorMin;
}
    //apply Method
Matrix Matrix::equal(Matrix const &m){
    row = m.row;
    col = m.col;
    for(int i=0;i<m.row;i++){
        for(int j=0;j<m.col;j++){
            array[i][j]=m.array[i][j];
        }
    }
}
Matrix Matrix::multiply(float const &value) //scalar multiplication: constant * matrix
{
	Matrix result(row, col);
    int i,j;
    
    for (i=0 ; i<row ; i++)
    {
        for (j=0 ; j<col ; j++)
        {
            result.array[i][j] = array[i][j] * value;
        }
    }
    return result;
}
 
Matrix Matrix::add(Matrix const &m) const // add 2 maxtrices
{
    assert(row==m.row && col==m.col);
    Matrix result(row, col);
    int i,j;

    for (i=0 ; i<row ; i++)
    {
        for (j=0 ; j<col ; j++)
        {
            result.array[i][j] = array[i][j] + m.array[i][j];
        }
    }
    return result;
}
 
Matrix Matrix::subtract(Matrix const &m) const //subtract 2 matrices
{
	assert(row==m.row && col==m.col);
    Matrix result(row, col);
    int i,j;
 
    for (i=0 ; i<row ; i++)
    {
        for (j=0 ; j<col ; j++)
        {
            result.array[i][j] = array[i][j] - m.array[i][j];
        }
    }
    return result;
}

Matrix Matrix::multiply(Matrix const &m) const //multiply element by element between 2 matrices
{
    assert(row==m.row && col==m.col);
    Matrix result(row, col);
    int i,j;
 
    for (i=0 ; i<row ; i++)
    {
        for (j=0 ; j<col ; j++)
        {
            result.array[i][j] = array[i][j] * m.array[i][j];
        }
    }
    return result;
}
 
Matrix Matrix::dot(Matrix const &m) const //dot product between 2 matrices
{
    assert(col==m.row);
 
    int i,j,h, mcol = m.col;
    float w=0;

    Matrix result(row, mcol);
 
    for (i=0 ; i<row ; i++)
    {
        for (j=0 ; j<mcol ; j++)
        {
            for (h=0 ; h<col ; h++)
            {
                w += array[i][h]*m.array[h][j];
            }
            result.array[i][j] = w;
            w=0;
        }
    }
    return result;
}
 
Matrix Matrix::transpose() const // ij element becomes ji element
{
    Matrix result(col, row);
    int i,j;
 
    for (i=0 ; i<col ; i++){
        for (j=0 ; j<row ; j++){
            result.array[i][j] = array[j][i];
        }
    }
    return result;
}
float Matrix::sumElementSquare(){
    assert(col==1);
    int i,j;
    //float temp;
    float cost=0;
    for (i=0;i<row;i++){
        cost += array[i][0]*array[i][0];
    }
    return cost;
}
 
Matrix Matrix::applyFunction(float (*function)(float)) const // takes as parameter a function 
                                            //which prototype looks like : double function(double x)
{
    Matrix result(row, col);
    int i,j;
 
    for (i=0 ; i<row ; i++)
    {
        for (j=0 ; j<col ; j++){
            result.array[i][j] = (*function)(array[i][j]);
        }
    }
    return result;
}

Matrix Matrix::mean(){ // find mean of the matrix
    float mean;
    float sum;
    Matrix result(row,col);
    for (int i=0;i<col;i++){
        mean=0;
        sum=0;
        for(int j=0;j<row;j++){
        sum += array[j][i];
        }
        mean=sum/row;
        result.array[0][i]=mean;
        vectorMean.push_back(mean);
    }
    //result.row =1;
    return result;
}
///////Normalize and Denormalize////////////////
Matrix Matrix::normalizeMinMax(){
    int i,j,k;
    float min,max;
    Matrix result(row,col);
    for (i=0;i<col;i++){
        //min= array[0][i];
        max= array[0][i];
        min= array[0][i];
        //find max and min of each Col
        for(j=0;j<row;j++){
            if (max <= array[j][i]){
                max = array[j][i];
            }
            if (min>= array[j][i]){
                min = array[j][i];
            }
            //if (min >= array[j][i]){min=array[j][i];}
        }
        //normalize 
        for(j=0;j<row;j++){
            //result.array[j][i]=(array[j][i]-min)/(max-min);
            //result.array[j][i]=(array[j][i]-result.mean().getArray()[0][i])/(max);
            result.array[j][i]=(array[j][i]-min)/(max-min);
        }
        vectorMax.push_back(max);
        vectorMin.push_back(min);
    }
    return result;
}
//Matrix Matrix::normalize(std::vector<float> mean,std::vector<float> max){
Matrix Matrix::normalizeMinMax(std::vector<float> min,std::vector<float> max){
    int i,j,k;
    Matrix result(row,col);
    for (i=0;i<col;i++){
        //normalize 
        for(j=0;j<row;j++){
            //result.array[j][i]=(array[j][i]-mean[i])/(max[i]);
            result.array[j][i]=(array[j][i]-min[i])/(max[i]-min[i]);
        }
    }
    return result;
}

//Matrix Matrix::deNormalize(std::vector<float>mean,std::vector<float>max){
Matrix Matrix::deNormalizeMinMax(std::vector<float> min,std::vector<float> max){
    int i,j;
    Matrix result(row,col);

    for (i=0;i<col;i++){
        //de-normalize 
        for(j=0;j<row;j++){
            result.array[j][i]=(array[j][i]*(max[i]-min[i]))+min[i];
        }
    }
    return result;
}

Matrix Matrix::normalizeMeanMax(){
    int i,j,k;
    float min,max;
    Matrix result(row,col);
    for (i=0;i<col;i++){
        max= array[0][i];
        //min= array[0][i];
        //find max and min of each Col
        for(j=0;j<row;j++){
            if (max <= array[j][i]){
                max = array[j][i];
            }
        }
        //normalize 
        for(j=0;j<row;j++){
            //result.array[j][i]=(array[j][i]-min)/(max-min);
            result.array[j][i]=(array[j][i]-this->mean().getArray()[0][i])/(max);
        }
        vectorMax.push_back(max);
    }
    return result;
}
//Matrix Matrix::normalize(std::vector<float> mean,std::vector<float> max){
Matrix Matrix::normalizeMeanMax(std::vector<float> mean,std::vector<float> max){
    int i,j,k;
    Matrix result(row,col);
    for (i=0;i<col;i++){
        //normalize 
        for(j=0;j<row;j++){
            result.array[j][i]=(array[j][i]-mean[i])/(max[i]);
        }
    }
    return result;
}

//Matrix Matrix::deNormalize(std::vector<float>mean,std::vector<float>max){
Matrix Matrix::deNormalizeMeanMax(std::vector<float> mean,std::vector<float> max){
    int i,j;
    Matrix result(row,col);

    for (i=0;i<col;i++){
        //de-normalize 
        for(j=0;j<row;j++){
            result.array[j][i]=(array[j][i]*max[i])+ mean[i];
        }
    }
    return result;
}

/////PRINT//////
void Matrix::print(std::ostream &flux) const // pretty print, taking into account the space between each element of the matrix
{
    int i,j;
    int maxLength[col] = {};
    std::stringstream ss;
 
    for (i=0 ; i<row ; i++)
    {
        for (j=0 ; j<col ; j++)
        {
            ss << array[i][j];
            if(maxLength[j] < ss.str().size())
            {
                maxLength[j] = ss.str().size();
            }
            ss.str(std::string());
        }
    }
 
    for (i=0 ; i<row ; i++)
    {
        for (j=0 ; j<col ; j++)
        {
            flux << array[i][j];
            ss << array[i][j];
            for (int k=0 ; k<maxLength[j]-ss.str().size()+1 ; k++)
            {
                flux << " ";
            }
            ss.str(std::string());
        }
        flux << std::endl;
    }
}
 
std::ostream& operator<<(std::ostream &flux, Matrix const &m)
{
    m.print(flux);
    return flux;
}
 