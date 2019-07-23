
#pragma once
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
#include <ctime>
#include "master_server.h"
#include "for_Mnist_MLP.h"
#include "for_Lenet5.h"
#include "client.h"
