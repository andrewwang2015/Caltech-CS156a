#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main()
  {
  mat A (1,3);
  A << 0.5 << 0.4 << 0.3 << endr;
  
  cout << A << endl;
  
  return 0;
  }