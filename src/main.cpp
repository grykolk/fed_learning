
#include"environment.h"

using namespace seal;
using namespace std;

void main() {

	char test_model;
	char test_mode;
	cout << "which model you want test" << endl;
	cout << "1. MLP for Mnist.\n2. MLP for Cifar_10. \n3. LeNet-5 for Mnist. \n4. FitNet-4 for Cifar-10" << endl;
	cin >> test_model;
	cout << "which mode you want run" << endl;
	cout << "1. test the model without federated.\n2. test the federated model without encryption. \n3. test the federated model with encryption." << endl;
	cin >> test_mode;
	cout << "|-------------------------------------------------|" << endl;
	cout << "|Tests will costs a lot of time, please be patient|" << endl;
	cout << "|-------------------------------------------------|" << endl;
	switch (test_model-'0') {
	case 1:
		run_Mnist_MLP(test_mode);
		break;
	case 2:
		run_Cifar10_MLP(test_mode);
		break;
	case 3: 
		run_LeNet_5_model(test_mode);
		break;
	case 4:
		run_FitNet4_model(test_mode);
		break;

	}


}
