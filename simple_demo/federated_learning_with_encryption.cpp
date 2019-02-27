#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <memory>
#include <limits>
#include<boost/python.hpp>
#include<boost/python/numpy.hpp>
#include "seal/seal.h"
//#ifdef _DEBUG
//#undef _DEBUG
//#include <python.h>
//#define _DEBUG
//#else
//#include <python.h>
//#endif
using namespace std;
using namespace seal;
void main() {
	char msg[256] = "11111 ";

	PyObject * pModule = NULL;
	PyObject * pFunc = NULL;
	PyObject * pArg = NULL;
	PyObject * run_learning = NULL;

	// 初始化python环境
	Py_Initialize();
	//PyRun_SimpleString("import sys"); //PyRun_SimpleString("import tensorflow as tf");
	// 导入python脚本
	pModule = PyImport_ImportModule("federated_mnistcnn");


	// 获得TensorFlow函数指针
	pFunc = PyObject_GetAttrString(pModule, "main");

	// 调用TensorFlow函数
	/*pArg = Py_BuildValue("(s)", "this is a call from c++");
	if (pModule != NULL) {
		PyEval_CallObject(pFunc, pArg);
	}*/
	run_learning=PyEval_CallObject(pFunc,NULL);
	Py_Finalize();



}

void pre_do() {

}