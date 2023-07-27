1. data文件夹中存放各类数据，包括backbone（runs=4）、全模型（runs=4）、用于建模的数据modling、用于建模的数据但是按照步长0.1采样的离散modeling_dis。文件夹中还包含一个`check_overhead.py`的文件，通过读取全模型数据，验证在同一batch时，不同config下的overhead基本相同，从而验证数据提取的正确性；
2. find_conv_num的目的是通过作差。逐步找出模型不同结构占用的conv2d的数量，最后的结果是53 + 38 + 20 + 20 = 131；
3. model_structure.txt文件中保存了全模型的结构图，以及所有config为0.5时的结构图，用来验证模型生成的正确性； 
4. `exp_run.py`是跑实验的主代码，其中可以设置运行次数，保存位置，config生成，数据提取与分配等；
5. `modling.py`用来构建不同module的延时预测器，并通过把数据分为训练集和预测集，验证预测的正确性；此外也增加了将全部38个config做为一组自变量从而预测模型整体延时的预测器，效果和单独预测每个config的预测器相当；现在只需要采用整体38个config建模，另外再加上模型peak memory的建模；
6. `py2onnx.py`用于把config指示下的torch模型转变为onnx格式的模型，用于mobile端和vpu上测试。