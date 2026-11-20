| model                               | precision           |   accuracy |   in_scope_acc |   oos_recall |   routing_acc |   threshold |   size_mb |   latency_ms |
|:------------------------------------|:--------------------|-----------:|---------------:|-------------:|--------------:|------------:|----------:|-------------:|
| tf-idf + LogReg                     | fp64                |      81.4  |          90.62 |         39.9 |         85.24 |        0.04 |      39.3 |         0.45 |
| FFN on avg. GloVe                   | fp32                |      79.38 |          88.89 |         36.6 |         84.15 |        0.16 |      22.8 |         0.15 |
| BERT-base (teacher)                 | fp32                |      90.67 |          95.56 |         68.7 |         92.53 |        0.18 |     438.4 |        32.18 |
| BERT-base INT8 (ONNX)               | INT8 (ONNX Runtime) |      89.73 |          95.33 |         64.5 |         91.82 |        0.14 |     110.4 |         7.64 |
| BERT-small distilled                | fp32                |      89.47 |          94.33 |         67.6 |         91.71 |        0.22 |     115.4 |         6.82 |
| BERT-small distilled INT8 (PyTorch) | INT8 (dynamic)      |      90.47 |          93.58 |         76.5 |         92.36 |        0.34 |      29.2 |         9.82 |
| BERT-small distilled INT8 (ONNX)    | INT8 (ONNX Runtime) |      89.2  |          94.6  |         64.9 |         91.44 |        0.18 |      29.1 |         1.7  |

Hardware: AMD Ryzen 7 3700X 8-Core Processor, 16 logical cores | Windows 10 | torch 2.15.0+cpu, onnxruntime 1.32.0 | 4 inference threads | CPU only
