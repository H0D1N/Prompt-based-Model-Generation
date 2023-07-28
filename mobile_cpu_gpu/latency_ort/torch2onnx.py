from model.utils import *

sample_num = 500
_, configs = configs_random(sample_num)
config_file = "onnx-models/configs.txt"

with open(config_file, "w") as f:
    for count, config in enumerate(configs):
        generate_onnx_model(config, count + 1)
        for config_data in config:
            f.write(str(config_data) + " ")
        f.write("\n")