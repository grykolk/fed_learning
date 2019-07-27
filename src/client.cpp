
#include"client.h"
using namespace seal;
using namespace std;
void run_Mnist_MLP(char mode) {
	int Total_Epoch = 0;
	int Banch_Size = 0;
	cout << "please input the total epoch to run" << endl;
	cin >> Total_Epoch;
	cout << "please input the banch size to run" << endl;
	cin >> Banch_Size;
	if (mode - '0' == 1) {//基准测试模式
		bool iid = 1;
		Client_Mnist_MLP client = Client_Mnist_MLP(1, Total_Epoch, 0, Banch_Size, iid);//客户端数量1，0轮通讯，iid数据
		client.Benchmark_without_federated();
	}
	if (mode - '0' == 2) {
		bool iid = 1;
		int C_round = 0;
		int client_number = 0;
		cout << "how many client in the federated cluster" << endl;
		cin >> client_number;
		cout << "which data type for federated learning" << endl;
		cout << "1. IID data\n2. non-IID" << endl;
		cin >> iid;
		cout << "how many Communication round during the trainning" << endl;
		cin >> C_round;
		Client_Mnist_MLP client = Client_Mnist_MLP(client_number, Total_Epoch, C_round, Banch_Size, iid);
		master_server *master = new master_server(client_number, 317600, 410);
		client.federated_without_encryption(master);
	}
	if (mode - '0' == 3) {
		bool iid = 1;
		int C_round = 0;
		int client_number = 0;
		int poly_modulus_degree = 0;
		int coeff_modulus = 0;
		cout << "how many client in the federated cluster" << endl;
		cin >> client_number;
		cout << "which data type for federated learning" << endl;
		cout << "1. IID data\n2. non-IID" << endl;
		cin >> iid;
		cout << "how many Communication round during the trainning" << endl;
		cin >> C_round;
		cout << "please set the poly modulus degree for encryption" << endl;
		cin >> poly_modulus_degree;
		cout << "please set the coeff_modulus for encryption" << endl;
		cin >> coeff_modulus;
		Client_Mnist_MLP client = Client_Mnist_MLP(client_number, Total_Epoch, C_round, Banch_Size, iid);
		master_server *master = new master_server(client_number, 317600, 410);
		master->set_EncryptionParameter(poly_modulus_degree, coeff_modulus);


		client.federated_with_encryption(poly_modulus_degree, coeff_modulus, master);
	}

}
void run_LeNet_5_model(char mode) {
	int Total_Epoch = 0;
	int Banch_Size = 0;
	cout << "please input the total epoch to run" << endl;
	cin >> Total_Epoch;
	cout << "please input the banch size to run" << endl;
	cin >> Banch_Size;
	if (mode - '0' == 1) {//基准测试模式
		bool iid = 1;
		Client_lenet_5 client = Client_lenet_5(1, Total_Epoch, 0, Banch_Size, iid);//客户端数量1，0轮通讯，iid数据
		client.Benchmark_without_federated();
	}
	if (mode - '0' == 2) {
		bool iid = 1;
		int C_round = 0;
		int client_number = 0;
		cout << "how many client in the federated cluster" << endl;
		cin >> client_number;
		cout << "which data type for federated learning" << endl;
		cout << "1. IID data\n2. non-IID" << endl;
		cin >> iid;
		cout << "how many Communication round during the trainning" << endl;
		cin >> C_round;
		Client_lenet_5 client = Client_lenet_5(client_number, Total_Epoch, C_round, Banch_Size, iid);
		master_server *master = new master_server(client_number, 61470, 236);
		client.federated_without_encryption(master);
	}
	if (mode - '0' == 3) {
		bool iid = 1;
		int C_round = 0;
		int client_number = 0;
		int poly_modulus_degree = 0;
		int coeff_modulus = 0;
		cout << "how many client in the federated cluster" << endl;
		cin >> client_number;
		cout << "which data type for federated learning" << endl;
		cout << "1. IID data\n2. non-IID" << endl;
		cin >> iid;
		cout << "how many Communication round during the trainning" << endl;
		cin >> C_round;
		cout << "please set the poly modulus degree for encryption" << endl;
		cin >> poly_modulus_degree;
		cout << "please set the coeff_modulus for encryption" << endl;
		cin >> coeff_modulus;
		Client_lenet_5 client = Client_lenet_5(client_number, Total_Epoch, C_round, Banch_Size, iid);
		master_server *master = new master_server(client_number, 61470, 236);
		master->set_EncryptionParameter(poly_modulus_degree, coeff_modulus);


		client.federated_with_encryption(poly_modulus_degree, coeff_modulus, master);
	}

}
void run_Cifar10_MLP(char mode) {
	int Total_Epoch = 0;
	int Banch_Size = 0;
	cout << "please input the total epoch to run" << endl;
	cin >> Total_Epoch;
	cout << "please input the banch size to run" << endl;
	cin >> Banch_Size;
	if (mode - '0' == 1) {//基准测试模式
		bool iid = 1;
		Client_Cifar10_MLP client = Client_Cifar10_MLP(1, Total_Epoch, 0, Banch_Size, iid);//客户端数量1，0轮通讯，iid数据
		client.Benchmark_without_federated();
	}
	if (mode - '0' == 2) {
		bool iid = 1;
		int C_round = 0;
		int client_number = 0;
		cout << "how many client in the federated cluster" << endl;
		cin >> client_number;
		cout << "which data type for federated learning" << endl;
		cout << "1. IID data\n2. non-IID" << endl;
		cin >> iid;
		cout << "how many Communication round during the trainning" << endl;
		cin >> C_round;
		Client_Cifar10_MLP client = Client_Cifar10_MLP(client_number, Total_Epoch, C_round, Banch_Size, iid);
		master_server *master = new master_server(client_number, 29401088, 8202);
		client.federated_without_encryption(master);
	}
	if (mode - '0' == 3) {
		bool iid = 1;
		int C_round = 0;
		int client_number = 0;
		int poly_modulus_degree = 0;
		int coeff_modulus = 0;
		cout << "how many client in the federated cluster" << endl;
		cin >> client_number;
		cout << "which data type for federated learning" << endl;
		cout << "1. IID data\n2. non-IID" << endl;
		cin >> iid;
		cout << "how many Communication round during the trainning" << endl;
		cin >> C_round;
		cout << "please set the poly modulus degree for encryption" << endl;
		cin >> poly_modulus_degree;
		cout << "please set the coeff_modulus for encryption" << endl;
		cin >> coeff_modulus;
		Client_Cifar10_MLP client = Client_Cifar10_MLP(client_number, Total_Epoch, C_round, Banch_Size, iid);
		master_server *master = new master_server(client_number, 29401088, 8202);
		master->set_EncryptionParameter(poly_modulus_degree, coeff_modulus);


		client.federated_with_encryption(poly_modulus_degree, coeff_modulus, master);
	}
}
void run_FitNet4_model(char mode) {
	int Total_Epoch = 0;
	int Banch_Size = 0;
	cout << "please input the total epoch to run" << endl;
	cin >> Total_Epoch;
	cout << "please input the banch size to run" << endl;
	cin >> Banch_Size;
	if (mode - '0' == 1) {//基准测试模式
		bool iid = 1;
		Client_FitNet4 client = Client_FitNet4(1, Total_Epoch, 0, Banch_Size, iid);//客户端数量1，0轮通讯，iid数据
		client.Benchmark_without_federated();
	}
	if (mode - '0' == 2) {
		bool iid = 1;
		int C_round = 0;
		int client_number = 0;
		cout << "how many client in the federated cluster" << endl;
		cin >> client_number;
		cout << "which data type for federated learning" << endl;
		cout << "1. IID data\n2. non-IID" << endl;
		cin >> iid;
		cout << "how many Communication round during the trainning" << endl;
		cin >> C_round;
		Client_FitNet4 client = Client_FitNet4(client_number, Total_Epoch, C_round, Banch_Size, iid);
		master_server *master = new master_server(client_number, 1069800, 1742);
		client.federated_without_encryption(master);
	}
	if (mode - '0' == 3) {
		bool iid = 1;
		int C_round = 0;
		int client_number = 0;
		int poly_modulus_degree = 0;
		int coeff_modulus = 0;
		cout << "how many client in the federated cluster" << endl;
		cin >> client_number;
		cout << "which data type for federated learning" << endl;
		cout << "1. IID data\n2. non-IID" << endl;
		cin >> iid;
		cout << "how many Communication round during the trainning" << endl;
		cin >> C_round;
		cout << "please set the poly modulus degree for encryption" << endl;
		cin >> poly_modulus_degree;
		cout << "please set the coeff_modulus for encryption" << endl;
		cin >> coeff_modulus;
		Client_FitNet4 client = Client_FitNet4(client_number, Total_Epoch, C_round, Banch_Size, iid);
		master_server *master = new master_server(client_number, 1069800, 1742);
		master->set_EncryptionParameter(poly_modulus_degree, coeff_modulus);


		client.federated_with_encryption(poly_modulus_degree, coeff_modulus, master);
	}
}