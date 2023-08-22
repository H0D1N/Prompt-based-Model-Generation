import onnx
import numpy as np
import onnxruntime as ort

ort_session = ort.InferenceSession("config_0.1_model.onnx", providers=["CPUExecutionProvider"])
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

input_data = np.array(np.random.random_sample((1, 3, 640, 640)), dtype=np.float32)
ort_inputs = {input_name: input_data}
ort_outs = ort_session.run([output_name], ort_inputs)
print(len(ort_outs))