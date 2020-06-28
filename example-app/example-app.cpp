
#include <torch/torch.h>
#include <iostream>
using namespace std;
int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  cout << tensor << endl;
  return 0;
}