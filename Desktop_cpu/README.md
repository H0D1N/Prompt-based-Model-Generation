1. data文件夹中存放各类数据，包括backbone（runs=4）、全模型（runs=4）、用于建模的数据modling、用于建模的数据但是按照步长0.1采样的离散modeling_dis。文件夹中还包含一个`check_overhead.py`的文件，通过读取全模型数据，验证在同一batch时，不同config下的overhead基本相同，从而验证数据提取的正确性；
2. find_conv_num的目的是通过作差。逐步找出模型不同结构占用的conv2d的数量，最后的结果是53 + 38 + 20 + 20 = 131；
3. model_structure.txt文件中保存了全模型的结构图，以及所有config为0.5时的结构图，用来验证模型生成的正确性； 
4. `latency_run.py` warmups = 2, runs = 3, sample2 = 3000 for batch 1, and 1000 for other batches(but we consider batch 1 now). Latency data is extracted into moduloes and configs can be gathered from all txt files.
5. `latency_modling.py`用来构建不同module的延时预测器，并通过把数据分为训练集和预测集，验证预测的正确性；此外也增加了将全部38个config做为一组自变量从而预测模型整体延时的预测器，效果和单独预测每个config的预测器相当；
6. `memory_run` warmups = 2, runs = 3, sample2 = 1000 for batch 1. Memory data is extracted into txt files which contain the table printed by torch.profiler. Configs data can be extracted from corresponding latency txt files which have the same meaning as latency_run.
7. `memory_modeling`采用整体38个config对模型peak memory建模；