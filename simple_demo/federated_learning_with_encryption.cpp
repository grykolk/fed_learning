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

	// 调用TensorFlow函数
	/*pArg = Py_BuildValue("(s)", "this is a call from c++");
	if (pModule != NULL) {
		PyEval_CallObject(pFunc, pArg);
	}*/
	PyObject * run_learning=PyEval_CallObject(pFunc,NULL);
	cout << "已取得数组，正在解析返回值";
	if (PyList_Check(run_learning)) {//解析python的返回值
		vector<double> get_data;
		int Index_i = 0, Index_k = 0, Index_m = 0, Index_n = 0;
		int size_of_list = PyList_Size(run_learning);//读取list的尺寸
		for (Index_i = 0;Index_i < size_of_list;Index_i++) {
			//读取List中的PyArrayObject对象，这里需要进行强制转换。
			PyArrayObject *ListItem = (PyArrayObject *)PyList_GetItem(run_learning, Index_i);
			int Rows = ListItem->dimensions[0], columns = ListItem->dimensions[1];
			for (Index_m = 0; Index_m < Rows; Index_m++) {

				for (Index_n = 0; Index_n < columns; Index_n++) {

					//get_data.push_back( *(double *)(ListItem->data + Index_m * ListItem->strides[0] + Index_n * ListItem->strides[1]));//访问数据，Index_m 和 Index_n 分别是数组元素的坐标，乘上相应维度的步长，即可以访问数组元素
				}
				cout << endl;
			}
			
		}
		cout << "解析完毕";


	}
	Py_Finalize();



}

void pre_do() {

}