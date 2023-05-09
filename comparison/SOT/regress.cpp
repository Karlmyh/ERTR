#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>
#include <time.h>
#include "SoftTree.cpp"

void readFromFile(string filename, vector< vector<double> > &X,
                  vector<double> &Y) {
  ifstream file;
    cout<<filename<<endl;
  file.open(filename.c_str());
  assert(file.is_open());
  
  double val;
  vector<double> row;
  
  while(!file.eof())
  {
    file >> val;
    
    if (file.peek() == '\n') { // if line ends, 'val' is the response value
      Y.push_back(val);
      X.push_back(row);
      row.clear();
    } else {                  // else, it is an entry of the feature vector
      row.push_back(val);
    }
  }
 
}


int main()
    {
    freopen( "../../results/realdata_tree/ST.txt", "w", stderr );
  
  // Initialize array of pointer

    
    char* dataset_list[20] = { "housing_scale","mpg_scale","airfoil","space_ga_scale","mg_scale","cpusmall_scale","aquatic","music", "redwine","ccpp","concrete","portfolio","building","yacht", "abalone","algerian","fish","communities","forestfires","cbm" };
 
    string dataset;
    // Printing Strings stored in 2D array
    for (int i = 0; i < 24; i++){
        dataset = dataset_list[i] ;
    
    

 
        
        cout<<dataset<<endl;

  

  //srand(time(NULL)); // random seed
  srand(123457);

  cout.precision(5);  // # digits after decimal pt
  cout.setf(ios::fixed,ios::floatfield);
    
   for (int iter = 0; iter < 20; iter++) {
       
       vector< vector< double> > X, V, U;
  vector<double> Y, R, T;
    
 
  string filename;
       
       
       if (iter <10){
    
    string prefix = "STdata/train_";
  filename = prefix
           + (char)(iter+'0')
           + '_'
           + dataset
           + ".txt";
  
  readFromFile(filename, X, Y);
 
    
prefix = "STdata/validate_";
  filename = prefix
           + (char)(iter+'0')
           + '_'
           + dataset
           + ".txt";
  
  
  readFromFile(filename, V, R);
  
  
    
prefix = "STdata/test_";
  filename = prefix
           + (char)(iter+'0')
           + '_'
           + dataset
           + ".txt";
    
  readFromFile(filename, U, T);
           
       }
       else{
       
    string prefix = "STdata/train_";
  filename = prefix
           + (char)('1')
           + (char)(iter-10+'0')
           + '_'
           + dataset
           + ".txt";
  
  readFromFile(filename, X, Y);
 
    
prefix = "STdata/validate_";
  filename = prefix
      + (char)('1')
           + (char)(iter-10+'0')
           + '_'
           + dataset
           + ".txt";
  
  
  readFromFile(filename, V, R);
  
  
    
prefix = "STdata/test_";
  filename = prefix
      + (char)('1')
           + (char)(iter-10+'0')
           + '_'
           + dataset
           + ".txt";
    
  readFromFile(filename, U, T);
           
       
       }
    

  clock_t time_start = clock() ;
    
    
  SoftTree st(X, Y, V, R);
  
  double y;
  double mse=0;
  
  //ofstream outf("out");
  
  mse = st.meanSqErr(X, Y);
  

  mse = st.meanSqErr(U, T);
  
  cout << "test_error: " << mse << "\t";
  cout << endl;
    
    
  clock_t time_end = clock() ;
    
  double duration = double(time_end - time_start)/CLOCKS_PER_SEC;
    
    cout<< "time:" << duration<<endl;
       
   
       cerr << dataset<<","<<mse<<","<<duration<<","<<iter << endl;
       
   }
    }
    
    
  return 0;
}

    /*
    char* dataset_list[24] = { "housing_scale","mpg_scale","airfoil","space_ga_scale","whitewine", "dakbilgic","mg_scale","bias","cpusmall_scale","aquatic","music", "redwine","ccpp","concrete","portfolio","building","yacht", "abalone","facebook","algerian","fish","communities","forestfires","cbm" };
    char* dataset_list[24] = { "housing_scale","mpg_scale","airfoil","space_ga_scale","mg_scale","cpusmall_scale","aquatic","music", "redwine","ccpp","concrete","portfolio","building","yacht", "abalone","algerian","fish","communities","forestfires","cbm" };
   
    */
