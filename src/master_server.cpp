#include"master_server.h"

using namespace seal;
using namespace std;
//class master_server::master_server {
//	vector<Ciphertext> weight;
//	Ciphertext biases;
//	vector<float> weight_without_encryption;
//	vector<float> biases_without_encryption;
//	int client_count = 0;
//	int weight_count = 0;
//	int biases_count = 0;
//	int poly_modulus_degree = 0;
//	int coeff_modulus = 0;
//	//Evaluator evaluator = NULL;
//public:
//
	master_server::master_server(int Client_count,int Weight_count,int Biases_count) {
		client_count = Client_count;
		weight_count = Weight_count;
		biases_count = Biases_count;
	}
	void master_server::set_EncryptionParameter(int Poly_modulus_degree,int Coeff_modulus ) {
		poly_modulus_degree = Poly_modulus_degree;
		coeff_modulus = Coeff_modulus;
	}
	void master_server::aggregate_with_encryption(vector <Ciphertext> Weight,vector <Ciphertext> Biases,int size) {
		weight.clear();
		EncryptionParameters parms = EncryptionParameters(scheme_type::CKKS);
		parms.set_poly_modulus_degree(poly_modulus_degree);
		parms.set_coeff_modulus(DefaultParams::coeff_modulus_128(coeff_modulus));
		auto context = SEALContext::Create(parms);
		Evaluator evaluator(context);
		for (int j = 0;j < size;j++) {
			weight.push_back(Weight[j]);
		}
		for (int i = 1;i < client_count;i++) {
			for (int j = 0;j <size;j++) {
				evaluator.add_inplace(weight[j],Weight[i * size + j]);
			}
			//evaluator.add_inplace(Biases[0], Biases[i]);
		}

		evaluator.add_many(Biases, biases);
	
	}
	void master_server::aggregate_without_encryption(vector<vector<float>> Weight,vector<vector<float>> Biases) {
		//auto a=sizeof(Weight);auto b = sizeof(Biases);
		//cout << "the size of total uploaded model: " << sizeof(Weight) + sizeof(float)*Weight.capacity()+sizeof(Biases)+ sizeof(float)*Biases.capacity() << endl;
		weight_without_encryption = vector<float>(Weight[0].size());//初始化数组
		biases_without_encryption = vector<float>(Biases[0].size());
		std::fill(weight_without_encryption.begin(),weight_without_encryption.end(),0);//填充0
		std::fill(biases_without_encryption.begin(), biases_without_encryption.end(), 0);//填充0
		for (int i = 0;i < client_count;i++) {
			for (int j = 0;j < weight_count;j++) {
				weight_without_encryption[j]+=Weight[i][j];

			}
			for (int j = 0;j < biases_count;j++) {
				biases_without_encryption[j]+=Biases[i][j];
			}
		}
		//cout <<"the size of updated model"<< sizeof(weight_without_encryption) + sizeof(float)*weight_without_encryption.capacity() + sizeof(biases_without_encryption) + sizeof(float)*biases_without_encryption.capacity() << endl;
	}
	vector<float> master_server::get_weight() {
		return weight_without_encryption;
	}
	vector<float> master_server::get_biases() {
		return biases_without_encryption;
	}
	vector<Ciphertext> master_server::get_encryption_weight() {
		return weight;
	}
	Ciphertext master_server::get_encryption_biases() {
		return biases;
	}
//};