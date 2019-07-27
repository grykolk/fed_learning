带加密的联邦学习
===========
在单机上模拟了联邦学习的架构，使用了FED_AVG作为聚合算法。
支持三种运行模式：单机模式，联邦模式和带加密的联邦模式。
---------------
demo仍然还在施工中<br>
Encryption_library文件夹里是CKKS算法的加密库和LICENSE<br>


|checklist|       | 
|---|---
|联邦学习最简单的demo for mnist|  done  | 
|测试加密库对于大数组的性能| done   | 
|调试最基本的C++调用python的方法|done   |    
|C++与python之间参数的传递| done    |
|用C++解析python的list+np多维数组| done   |
|加密模块+计算模块| done      |
|把计算好的数据重构为list+np多维数组| done     |
|把数据传回python继续进行学习|done      | 
|多层感知网络MLP+联邦加密学习|done      |
|LeNet5联邦加密学习|done      |
|FitNet4联邦加密学习|done       |
|测试性能| working    |
|把demo做成可以在多个电脑上实机运行的成品（有时间的话)|not yet |
