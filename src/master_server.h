#pragma once

#ifndef MASTER_SERVER
#define MASTER_SERVER
#include <vector>
#include "seal/seal.h"
class master_server {
	std::vector<seal::Ciphertext> weight;
	seal::Ciphertext biases;
	std::vector<float> weight_without_encryption;
	std::vector<float> biases_without_encryption;
	int client_count;
	int weight_count;
	int biases_count;
	int poly_modulus_degree;
	int coeff_modulus;
public:
	master_server(int Client_count, int Weight_count, int Biases_count);
	void set_EncryptionParameter(int Poly_modulus_degree = 0, int Coeff_modulus = 0);
	void aggregate_with_encryption(std::vector <seal::Ciphertext> Weight, std::vector <seal::Ciphertext> Biases, int size);
	void aggregate_without_encryption(std::vector<std::vector<float>> Weight, std::vector<std::vector<float>> Biases);
	std::vector<float> get_weight();
	std::vector<float> get_biases();
	std::vector<seal::Ciphertext> get_encryption_weight();
	seal::Ciphertext get_encryption_biases();
};
#endif // !1
