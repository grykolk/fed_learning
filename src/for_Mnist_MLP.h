#pragma once
#ifndef FOR_MNIST_MLP
#define FOR_MNIST_MLP
#include "environment.h"
using namespace seal;
using namespace std;
class Mnist_MLP {
private:
	PyObject * pModule;
	PyObject * pFunc;
	PyObject * pArg;
public:
	int clinet_num;
	int Epoch;
	vector <vector<float>> weight_2D;
	vector <vector<float>> biases_2D;
	Mnist_MLP();
	Mnist_MLP(int Clinet_num);
	void set_parameter(int client_num, int iid_data, int epoch, int Banch_Size);
	void benchmark();
	void training(vector<float> Weight, vector<float> Biases);
	void training_with_initialization();
	double test_accuracy(vector<float> Weight, vector<float> Biases);
	void draw_plt(vector <double> acc, int Communication_Epoch);
	void get_parameter(PyObject * list1);
	int init_numpy();
};
class Client_Mnist_MLP {
private:
	int client_num;
	int Total_Epoch;
	int Communication_Epoch;
	bool IID_Data;
	int Banch_Size;
	Mnist_MLP mnist_mlp;
	vector<float> updated_Weight;
	vector<float> updated_Biases;
	vector<double> accuracy;
public:
	Client_Mnist_MLP(int Client_num, int total_epoch, int communication_epoch, int banch_size, bool iid_data);
	void Benchmark_without_federated();
	void federated_without_encryption(master_server *master);
	void federated_with_encryption(int poly_modulus_degree, int coeff_modulus, master_server *master);
};
#endif // !FOR_MNIST_MLP
