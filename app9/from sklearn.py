from sklearn.linear_model import LinearRegression
import numpy as np
X = np.random.rand(1000, 1)
y = 3*X +5
model = LinearRegression()
model.fit(X, y)
from skl2onnx import to_onnx
onnx_model = to_onnx(model, X)
with open("model.onnx", "wb") as f:
 f.write(onnx_model.SerializeToString())
import onnxruntime as ort
sess = ort.InferenceSession("model.onnx")
import time
t0 = time.time()
for _ in range(10000):
 model.predict(X)
sktime = time.time() - t0
t0 = time.time()
for _ in range(10000):
 sess.run(None, {"X":X})
onnxTime = time.time()-t0
print("Scikit-Learn time", sktime)
print("ONNX time", onnxTime)