from onnx_tf.backend import prepare
import onnx
import tensorflow as tf

sample_num = 1000
for count in range(sample_num):
    ONNX_PATH = "../onnx_models_op12/" + str(count + 1) + ".onnx"
    TF_PATH = "../tf_models/" + str(count + 1)
    onnx_model = onnx.load(ONNX_PATH)  # load onnx model
    tf_rep = prepare(onnx_model)  # creating TensorflowRep object
    tf_rep.export_graph(TF_PATH)
    TFLITE_PATH = "../tflite_models/" + str(count + 1) + ".tflite"
    converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tf_lite_model = converter.convert()
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tf_lite_model)