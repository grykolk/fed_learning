#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <vector>
#include <thread>
#include <mutex>
#include <memory>
#include <limits>
#include <site-packages/numpy/core/include/numpy/arrayobject.h>
#include "seal/seal.h"



using namespace std;
using namespace seal;
int init_numpy() {//初始化 numpy 执行环境，主要是导入包，python2.7用void返回类型，python3.0以上用int返回类型

	import_array();
}
void main() {
	char msg[256] = "11111 ";

	PyObject * pModule = NULL;
	PyObject * pFunc = NULL;
	PyObject * pArg = NULL;

	// 初始化python环境
	Py_Initialize();
	init_numpy();//初始化numpy环境
	// 导入python脚本
	pModule = PyImport_ImportModule("federated_mnistcnn");


	// 获得TensorFlow函数指针
	pFunc = PyObject_GetAttrString(pModule, "main");
	//pFunc = PyObject_GetAttrString(pModule, "test");
	// 调用TensorFlow函数
	/*pArg = Py_BuildValue("(s)", "this is a call from c++");
	if (pModule != NULL) {
		PyEval_CallObject(pFunc, pArg);
	}*/
	PyObject * list1=PyEval_CallObject(pFunc,NULL);
	cout << "已取得数组，正在解析返回值";
	if (PyList_Check(list1)) {//解析python的返回值
		vector<double> get_data;
		int Index_1 = 0, Index_2 = 0, Index_3 = 0;//第一层（权重，偏移量）。第二层（每个客户端的数据）。第三层（四个网络）
		int size_of_list = PyList_Size(list1);//读取list的尺寸
		for (Index_1 = 0;Index_1 < size_of_list;Index_1++) {
			if(Index_1==0){
			//剥开第一层list
			//读取List中的PyArrayObject对象，这里需要进行强制转换。
			//PyArrayObject *ListItem = (PyArrayObject *)PyList_GetItem(run_learning, Index_1);
			PyObject *List2 = (PyObject *)PyList_GetItem(list1, Index_1);//剥开第二层list
			for (Index_2 = 0;Index_2 < PyList_Size(List2);Index_2++) {
				PyObject *List3 = (PyObject *)PyList_GetItem(List2, Index_2);//剥开第三层list.numpy数组的维度=第零层5，5，1，32第一层5，5，32，64第二层3136, 128第三层（128，10）
				for (Index_3 = 0;Index_3 < PyList_Size(List3);Index_3++) {
					PyArrayObject *data = (PyArrayObject *)PyList_GetItem(List3, Index_3);
					switch (Index_3) {
					case 0:
						for (int i = 0;i < 5;i++) {
							for (int j = 0;j < 5;j++) {
								for (int m = 0;m < 1;m++) {
									for (int n = 0;n < 32;n++) {
										cout << *(float *)(data->data + i * data->strides[0] + j * data->strides[1] + m * data->strides[2] + n * data->strides[3]) << endl;
									}
								}
							}
						}
						break;
					case 1:
						for (int i = 0;i < 5;i++) {
							for (int j = 0;j < 5;j++) {
								for (int m = 0;m < 32;m++) {
									for (int n = 0;n < 64;n++) {
										cout << *(float *)(data->data + i * data->strides[0] + j * data->strides[1] + m * data->strides[2] + n * data->strides[3]);
									}
								}
							}
						}
						break;
					case 2:
						for (int i = 0;i < 3136;i++) {
							for (int j = 0;j < 128;j++) {
								cout << *(float *)(data->data + i * data->strides[0] + j * data->strides[1]);
							}
						}

						break;
					case 3:
						for (int i = 0;i < 128;i++) {
							for (int j = 0;j < 10;j++) {
								cout << *(float *)(data->data + i * data->strides[0] + j * data->strides[1]);
							}
						}
						break;


					}
				}
			}
			if (Index_1 == 1) {
				PyObject *List2 = (PyObject *)PyList_GetItem(list1, Index_1);//剥开第二层list#第二个数组[100,4,32]
				for (Index_2 = 0;Index_2 < PyList_Size(List2);Index_2++) {
					PyObject *List3 = (PyObject *)PyList_GetItem(List2, Index_2);
					for (Index_3 = 0;Index_3 < PyList_Size(List3);Index_3++) {
						PyArrayObject *data = (PyArrayObject *)PyList_GetItem(List3, Index_3);
						for (int i = 0;i < 32;i++) {
							cout << *(float *)(data->data + i * data->strides[0] );
						}
					}
				}
			}
			//	int Rows = ListItem->dimensions[0], columns = ListItem->dimensions[1];
			//	for (int Index_m = 0; Index_m < Rows; Index_m++) {

			//		for (int Index_n = 0; Index_n < columns; Index_n++) {

			//			//get_data.push_back( *(double *)(ListItem->data + Index_m * ListItem->strides[0] + Index_n * ListItem->strides[1]));//访问数据，Index_m 和 Index_n 分别是数组元素的坐标，乘上相应维度的步长，即可以访问数组元素
			//		}
			//		cout << endl;
			//	}
			}

		}
		cout << "解析完毕";


	}
	Py_Finalize();



}

void pre_do() {

}