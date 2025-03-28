#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#include "Perceptron.h"

using namespace std;

int main() {
  const int epochs = 10000;
  const float learning_rate = 0.1f;

  vector<vector<float>> X;
  vector<float> Y;
  for (int i = 0; i < 100; i++) {
    float a = rand() / (RAND_MAX + 1.0);
    float b = rand() / (RAND_MAX + 1.0);
    X.push_back( {a, b} );
    Y.push_back(a+b);
  }

  Neuron hidden1(2);
  Neuron hidden2(2);
  Neuron output_neuron(2);

  train(X, Y, hidden1, hidden2, output_neuron, epochs, learning_rate);

  // --- Inference after training ---
  cout << "\nTrained SUM:\n";
  for (size_t i = 0; i < X.size(); ++i) {
    float h1_output = hidden1.forward(X[i]);
    float h2_output = hidden2.forward(X[i]);
    float prediction = output_neuron.forward({h1_output, h2_output});

    printf("%8.5f + %8.5f = %8.5f (%8.5f)\n", X[i][0], X[i][1], prediction, Y[i]);
  }

  return 0;
}
