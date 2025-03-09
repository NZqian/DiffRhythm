from onnxruntime import InferenceSession, GraphOptimizationLevel, SessionOptions
options = SessionOptions()  # initialize session options
options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

session = InferenceSession(
    "g2p/sources/g2p_chinese_model/poly_bert_model.onnx",
    sess_options=options,
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],  # CPUExecutionProvider #CUDAExecutionProvider
)
print(session)