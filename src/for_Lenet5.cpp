
#include"for_Lenet5.h"

using namespace seal;
using namespace std;
//class Lenet_5::Lenet_5 {
//private:
//	PyObject * pModule = NULL;
//	PyObject * pFunc = NULL;
//	PyObject * pArg = NULL;
//public:
//	int clinet_num = NULL;
//	int Epoch = 0;
//	vector <vector<float>> weight_2D;
//	vector <vector<float>> biases_2D;
//	//vector<float> weight;
//	//vector<float> biases;
//
Lenet_5::Lenet_5(){}
	Lenet_5::Lenet_5(int Clinet_num) {
		// 初始化python环境
		Py_Initialize();
		init_numpy();//初始化numpy环境
		// 导入python脚本
		pModule = PyImport_ImportModule("LeNet5");
		clinet_num = Clinet_num;
	}
	void Lenet_5::set_parameter(int client_num, int iid_data, int epoch,int Banch_Size) {
		Epoch = epoch;
		PyObject *ArgList = PyTuple_New(4);
		PyTuple_SetItem(ArgList, 0, PyLong_FromLong(client_num));
		PyTuple_SetItem(ArgList, 1, PyLong_FromLong(iid_data));
		PyTuple_SetItem(ArgList, 2, PyLong_FromLong(epoch));
		PyTuple_SetItem(ArgList, 3, PyLong_FromLong(Banch_Size));
		pFunc = PyObject_GetAttrString(pModule, "init");
		PyEval_CallObject(pFunc, ArgList);

	}
	void Lenet_5::benchmark() {
		pFunc = PyObject_GetAttrString(pModule, "Benchmark");
		PyObject * list1 = PyObject_CallObject(pFunc, NULL);
	}
	void Lenet_5::training(vector<float> Weight, vector<float> Biases) {
		PyObject *ArgList = PyTuple_New(2);//定义python的输入值
		PyObject *PyList_weight = PyList_New(61470);//定义权重list
		PyObject *PyList_biase = PyList_New(236);//定义偏置list
		for (int Index_i = 0; Index_i < PyList_Size(PyList_weight); Index_i++) {

			PyList_SetItem(PyList_weight, Index_i, PyFloat_FromDouble(Weight[Index_i]));//权重层赋值
		}
		for (int Index_i = 0; Index_i < PyList_Size(PyList_biase); Index_i++) {

			PyList_SetItem(PyList_biase, Index_i, PyFloat_FromDouble(Biases[Index_i]));//偏置层赋值
		}
		PyTuple_SetItem(ArgList, 0, PyList_weight);
		PyTuple_SetItem(ArgList, 1, PyList_biase);
		pFunc = PyObject_GetAttrString(pModule, "next");
		PyObject * list1 = PyObject_CallObject(pFunc, ArgList);
		get_parameter(list1);
	}
	void Lenet_5::training_with_initialization() {
		pFunc = PyObject_GetAttrString(pModule, "run");
		PyObject * list1 = PyEval_CallObject(pFunc, NULL);
		get_parameter(list1);
	}
	double Lenet_5::test_accuracy(vector<float> Weight, vector<float> Biases) {
		PyObject *ArgList = PyTuple_New(2);//定义python的输入值
		PyObject *PyList_weight = PyList_New(61470);//定义权重list
		PyObject *PyList_biase = PyList_New(236);//定义偏置list
		for (int Index_i = 0; Index_i < PyList_Size(PyList_weight); Index_i++) {

			PyList_SetItem(PyList_weight, Index_i, PyFloat_FromDouble(Weight[Index_i]));//权重层赋值
		}
		for (int Index_i = 0; Index_i < PyList_Size(PyList_biase); Index_i++) {

			PyList_SetItem(PyList_biase, Index_i, PyFloat_FromDouble(Biases[Index_i]));//偏置层赋值
		}
		PyTuple_SetItem(ArgList, 0, PyList_weight);
		PyTuple_SetItem(ArgList, 1, PyList_biase);
		pFunc = PyObject_GetAttrString(pModule, "test");
		PyObject * list1=PyObject_CallObject(pFunc, ArgList);
		//PyList_Check(list1);
		//double a=PyFloat_AsDouble(list1);
		return PyFloat_AsDouble(list1);
		//return 0.1;
	}
	void Lenet_5::draw_plt(vector <double> acc, int Communication_Epoch) {
		PyObject *ArgList = PyTuple_New(3);//定义python的输入值
		PyObject *para1 = PyList_New(acc.size());
		PyObject *para2 = PyList_New(Communication_Epoch);
		for (int Index_i = 0; Index_i < PyList_Size(para1); Index_i++) {

			PyList_SetItem(para1, Index_i, PyFloat_FromDouble(acc[Index_i]));
		}
		for (int Index_i = 0; Index_i < PyList_Size(para2); Index_i++) {

			PyList_SetItem(para2, Index_i, PyFloat_FromDouble(Index_i*Epoch));
		}
		PyObject *para3 = Py_BuildValue("f", 3.14);
		pFunc = PyObject_GetAttrString(pModule, "draw_plt");
		PyTuple_SetItem(ArgList, 0, para2);
		PyTuple_SetItem(ArgList, 1, para1);
		PyTuple_SetItem(ArgList, 2, para3);
		PyObject_CallObject(pFunc, ArgList);

	}
	void Lenet_5::get_parameter(PyObject * list1) {
		if (PyList_Check(list1)) {//解析python的返回值
			vector<float> vector_data;
			weight_2D.clear();
			biases_2D.clear();
			int Index_1 = 0, Index_2 = 0, Index_3 = 0;//第一层（权重，偏移量）。第二层（每个客户端的数据）。第三层（两层网络）
			int size_of_list = PyList_Size(list1);//读取list的尺寸
			for (Index_1 = 0;Index_1 < size_of_list;Index_1++) {
				if (Index_1 == 0) {
					//剥开第一层list
					//读取List中的PyArrayObject对象，这里需要进行强制转换。
					//PyArrayObject *ListItem = (PyArrayObject *)PyList_GetItem(run_learning, Index_1);
					PyObject *List2 = (PyObject *)PyList_GetItem(list1, Index_1);//剥开第二层list
					for (Index_2 = 0;Index_2 < PyList_Size(List2);Index_2++)
					{
						vector_data.clear();//清除临时向量数据
						PyObject *List3 = (PyObject *)PyList_GetItem(List2, Index_2);//剥开第三层list.numpy数组的维度=第零层（5，5，1，6）第一层（5，5，6，16）第二层（5，5，16，120）第三层（120，84）第四层（84，10）
						for (Index_3 = 0;Index_3 < PyList_Size(List3);Index_3++) {
							PyArrayObject *data = (PyArrayObject *)PyList_GetItem(List3, Index_3);
							switch (Index_3) {
							case 0:
								for (int i = 0;i < 5;i++) {
									for (int j = 0;j < 5;j++) {
										for (int x = 0;x < 1;x++) {
											for (int y = 0;y < 6;y++) {
												vector_data.push_back(*(float *)(data->data + i * data->strides[0] + j * data->strides[1] + x * data->strides[2] + y * data->strides[3]));
											}
										}
									}
								}
								break;
							case 1:
								for (int i = 0;i < 5;i++) {
									for (int j = 0;j < 5;j++) {
										for (int x = 0;x < 6;x++) {
											for (int y = 0;y < 16;y++) {
												vector_data.push_back(*(float *)(data->data + i * data->strides[0] + j * data->strides[1] + x * data->strides[2] + y * data->strides[3]));
											}
										}
									}
								}
								break;
							case 2:
								for (int i = 0;i < 5;i++) {
									for (int j = 0;j < 5;j++) {
										for (int x = 0;x < 16;x++) {
											for (int y = 0;y < 120;y++) {
												vector_data.push_back(*(float *)(data->data + i * data->strides[0] + j * data->strides[1] + x * data->strides[2] + y * data->strides[3]));
											}
										}
									}
								}
								break;
							case 3:
								for (int i = 0;i < 120;i++) {
									for (int j = 0;j < 84;j++) {

										vector_data.push_back(*(float *)(data->data + i * data->strides[0] + j * data->strides[1]));

									}
								}
								break;
							case 4:
								for (int i = 0;i < 84;i++) {
									for (int j = 0;j < 10;j++) {

										vector_data.push_back(*(float *)(data->data + i * data->strides[0] + j * data->strides[1]));

									}
								}
								break;
							}
						}
						weight_2D.push_back(vector_data);


					}
				}
				if (Index_1 == 1) {

					PyObject *List2 = (PyObject *)PyList_GetItem(list1, Index_1);//位移的维度第6,16,120,84,10
					for (Index_2 = 0;Index_2 < PyList_Size(List2);Index_2++) {
						PyObject *List3 = (PyObject *)PyList_GetItem(List2, Index_2);
						vector_data.clear();//清除临时向量数据
						for (Index_3 = 0;Index_3 < PyList_Size(List3);Index_3++) {
							PyArrayObject *data = (PyArrayObject *)PyList_GetItem(List3, Index_3);
							switch (Index_3) {
							case 0:
								for (int i = 0;i < 6;i++) {
									vector_data.push_back(*(float *)(data->data + i * data->strides[0]));
								}
								break;
							case 1:
								for (int i = 0;i < 16;i++) {
									vector_data.push_back(*(float *)(data->data + i * data->strides[0]));
								}
								break;
							case 2:
								for (int i = 0;i < 120;i++) {
									vector_data.push_back(*(float *)(data->data + i * data->strides[0]));
								}
								break;
							case 3:
								for (int i = 0;i < 84;i++) {
									vector_data.push_back(*(float *)(data->data + i * data->strides[0]));
								}
								break;
							case 4:
								for (int i = 0;i < 10;i++) {
									vector_data.push_back(*(float *)(data->data + i * data->strides[0]));
								}
								break;
							}
						}
						biases_2D.push_back(vector_data);
					}
				}
			}

		}

	}
	int Lenet_5::init_numpy() {//初始化 numpy 执行环境，主要是导入包，python2.7用void返回类型，python3.0以上用int返回类型

		import_array();
	}
//};
//class Client_lenet_5::Client_lenet_5 {
//private:
//	int client_num;
//	int Total_Epoch = 0;
//	int Communication_Epoch = 0;
//	bool IID_Data = NULL;
//	int Banch_Size = 0;
//	Lenet_5 lenet5 = NULL;
//	vector<float> updated_Weight;
//	vector<float> updated_Biases;
//	vector<double> accuracy;
//
//public:
	Client_lenet_5::Client_lenet_5(int Client_num,int total_epoch,int communication_epoch,int banch_size,bool iid_data) {
		client_num = Client_num;
		Total_Epoch = total_epoch;
		Communication_Epoch = communication_epoch;
		IID_Data = iid_data;
		Banch_Size = banch_size;
		lenet5 =Lenet_5(client_num);
		
	}
	void Client_lenet_5::Benchmark_without_federated() {
		lenet5.set_parameter(client_num,1,Total_Epoch, Banch_Size);
		lenet5.benchmark();
	}
	void Client_lenet_5::federated_without_encryption(master_server *master) {
		int E = Communication_Epoch;
		while (E > 0) {
			if (E == Communication_Epoch) {
				lenet5.set_parameter(client_num, IID_Data, Total_Epoch/ Communication_Epoch, Banch_Size);
				lenet5.training_with_initialization();
				master->aggregate_without_encryption(lenet5.weight_2D, lenet5.biases_2D);

				accuracy.push_back(lenet5.test_accuracy(master->get_weight(), master->get_biases()));
			}
			else {
				lenet5.training(master->get_weight(), master->get_biases());
				master->aggregate_without_encryption(lenet5.weight_2D, lenet5.biases_2D);
				accuracy.push_back(lenet5.test_accuracy(master->get_weight(), master->get_biases()));
			}
		E -= 1;
		}

		lenet5.draw_plt(accuracy, Communication_Epoch);
	}
	void Client_lenet_5::federated_with_encryption(int poly_modulus_degree, int coeff_modulus,master_server *master) {
		EncryptionParameters parms = EncryptionParameters(scheme_type::CKKS);
		parms.set_poly_modulus_degree(poly_modulus_degree);
		parms.set_coeff_modulus(DefaultParams::coeff_modulus_128(coeff_modulus));
		auto context = SEALContext::Create(parms);
		KeyGenerator keygen(context);
		auto public_key = keygen.public_key();
		auto secret_key = keygen.secret_key();
		auto relin_keys = keygen.relin_keys(DefaultParams::dbc_max());
		//创建加密实例
		Encryptor encryptor(context, public_key);
		Decryptor decryptor(context, secret_key);
		CKKSEncoder encoder(context);
		size_t slot_count = encoder.slot_count();
		//缩放比例
		double scale = pow(2.0, 60);
		int E = Communication_Epoch;
		while (E > 0) {
			if (E == Communication_Epoch) {
				lenet5.set_parameter(client_num, IID_Data, Total_Epoch / Communication_Epoch, Banch_Size);
				lenet5.training_with_initialization();
				cout << "加密" << endl;
				//加密
				vector <Ciphertext> Cweight;
				vector <Ciphertext> Cbiases;
				Plaintext Ptemp;
				Ciphertext Ctemp;
				//vector<double> temp;vector<float> temp1;
				for (int i = 0;i < client_num;i++) {
					//temp1 = lenet5.weight_2D[i];
					for (int j = 0;j < 30;j++) {//一个密文最大容量是2048，总参数数量是61470，61470//2048+1=30

						//temp = vector<double>(temp1.begin() + i * 2048, temp1.begin() + i * 2048 + 2048);
						encoder.encode(vector<double>(lenet5.weight_2D[i].begin() + j * 2048, lenet5.weight_2D[i].begin() + j * 2048 + 2048), scale, Ptemp);
						encryptor.encrypt(Ptemp, Ctemp);
						Cweight.push_back(Ctemp);

					}
					encoder.encode(vector<double>(lenet5.weight_2D[i].end() - 30, lenet5.weight_2D[i].end()), Ctemp.parms_id(), Ctemp.scale(), Ptemp);//第31组权重
					encryptor.encrypt(Ptemp, Ctemp);
					Cweight.push_back(Ctemp);
					//加密biases
					encoder.encode(vector<double>(lenet5.biases_2D[i].begin(), lenet5.biases_2D[i].end()), Ctemp.parms_id(), Ctemp.scale(), Ptemp);
					encryptor.encrypt(Ptemp, Ctemp);
					Cbiases.push_back(Ctemp);

				}
				//聚合
				cout << "聚合" << endl;
				master->aggregate_with_encryption(Cweight, Cbiases,31);
				Cweight.clear();
				Ciphertext C_biases = master->get_encryption_biases();
				Cweight = master->get_encryption_weight();
				//解密
				cout << "解密" << endl;
				vector<double>temp_vector(2048);
				vector<double>Weight;vector<double> Biases;
				for (int i = 0;i < 31;i++) {
					decryptor.decrypt(Cweight[i], Ptemp);
					encoder.decode(Ptemp, temp_vector);
					Weight.insert(Weight.end(), temp_vector.begin(), temp_vector.end());

				}
				decryptor.decrypt(C_biases, Ptemp);
				encoder.decode(Ptemp, Biases);
				updated_Weight = vector<float>(Weight.begin(), Weight.end());
				updated_Biases = vector<float>(Biases.begin(), Biases.end());
				accuracy.push_back(lenet5.test_accuracy(updated_Weight,updated_Biases));
			}
			else {
				lenet5.training(updated_Weight, updated_Biases);
				//加密
				cout << "加密" << endl;
				vector <Ciphertext> Cweight;
				vector <Ciphertext> Cbiases;
				Plaintext Ptemp;
				Ciphertext Ctemp;
				for (int i = 0;i < client_num;i++) {
					for (int j = 0;j < 30;j++) {//一个密文最大容量是2048，总参数数量是61470，61470//2048+1=30
						encoder.encode(vector<double>(lenet5.weight_2D[i].begin() + j * 2048, lenet5.weight_2D[i].begin() + j * 2048 + 2048), scale, Ptemp);
						encryptor.encrypt(Ptemp, Ctemp);
						Cweight.push_back(Ctemp);

					}
					encoder.encode(vector<double>(lenet5.weight_2D[i].end() - 30, lenet5.weight_2D[i].end()), Ctemp.parms_id(), Ctemp.scale(), Ptemp);//第31组权重
					encryptor.encrypt(Ptemp, Ctemp);
					Cweight.push_back(Ctemp);
					//加密biases
					encoder.encode(vector<double>(lenet5.biases_2D[i].begin(), lenet5.biases_2D[i].end()), Ctemp.parms_id(), Ctemp.scale(), Ptemp);
					encryptor.encrypt(Ptemp, Ctemp);
					Cbiases.push_back(Ctemp);

				}
				//聚合
				cout << "聚合" << endl;
				master->aggregate_with_encryption(Cweight, Cbiases, 31);
				Cweight.clear();
				Ciphertext C_biases = master->get_encryption_biases();
				Cweight = master->get_encryption_weight();
				//解密
				cout << "解密" << endl;
				vector<double>temp_vector(2048);
				vector<double>Weight;vector<double> Biases;
				for (int i = 0;i < 31;i++) {
					decryptor.decrypt(Cweight[i], Ptemp);
					encoder.decode(Ptemp, temp_vector);
					Weight.insert(Weight.end(), temp_vector.begin(), temp_vector.end());

				}
				decryptor.decrypt(C_biases, Ptemp);
				encoder.decode(Ptemp, Biases);

				updated_Weight = vector<float>(Weight.begin(), Weight.end());
				updated_Biases = vector<float>(Biases.begin(), Biases.end());
				accuracy.push_back(lenet5.test_accuracy(updated_Weight, updated_Biases));
			}
			E -= 1;
		}

		lenet5.draw_plt(accuracy, Communication_Epoch);


	}

//};
