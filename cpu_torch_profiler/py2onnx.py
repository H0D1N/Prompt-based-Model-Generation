from model. utils import *
import torch
import torch.onnx
import onnxruntime as ort
configs = generate_configs(38, 0.1, 1.01, 0.1)
save_dir = "onnx_models/"

for config in configs:
    model = get_model(config)
    model.eval()
    inputs = torch.randn(1, 3, 640, 640, dtype=torch.float)
    torch.onnx.export(
        model, inputs, save_dir + "config_" + str(round(config[0], 1)) + "_model"  + ".onnx", verbose=True, input_names=["inputs"], output_names=["outputs"])

# sess = ort.InferenceSession("onnx_models/config_0.1_model.onnx")
# input_name = sess.get_inputs()[0].name
# print(input_name)
# output_name = sess.get_outputs()[0].name
# print(output_name)
# inputs = torch.randn(1, 3, 640, 640, dtype=torch.float)
# ort_inputs = {input_name: inputs.detach().cpu().numpy()}
# ort_outs = sess.run([output_name], ort_inputs)
# print(ort_outs[0].shape)

