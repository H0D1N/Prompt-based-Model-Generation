from model. utils import *
import torch
import torch.onnx
import numpy as np
# configs = generate_configs(38, 0.1, 1.01, 0.1)
_, configs = configs_random(1)

for config in configs:
    model = get_model(config)
    model.eval()
    inputs = torch.randn(1, 3, 640, 640, dtype=torch.float, requires_grad=True)
    torch_out = model(inputs)
    
    file_path = "config_" + str(round(config[0], 1)) + "_model"  + ".onnx"
    # inputs = inputs.detach().cpu().numpy() if inputs.requires_grad else inputs.cpu().numpy()
    torch.onnx.export(
        model, 
        inputs, 
        file_path, 
        export_params=True, 
        do_constant_folding=True,
        verbose=True, 
        opset_version=12,
        input_names = ['input'],   # the model's input names
        output_names = ['output'] # the model's output names
    )

    # print(len(model(inputs)[0]))
# print(len(torch_out), len(torch_out[0]), len(torch_out[0][0]), len(torch_out[0][0][0]), len(torch_out[0][0][0][0]), len(torch_out[0][0][0][0][0]))
import onnxruntime

ort_session = onnxruntime.InferenceSession(file_path, providers=['CPUExecutionProvider'])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(torch_out[0].detach().numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
