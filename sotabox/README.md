# sotabox

sotabox is machine learning & AI tools including state-of-the-art algorithms.

### model_util

``` python
import tensorflow as tf
from sotabox.model_util import extract_weights

tf.enable_eager_execution()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

weights = extract_weights(model)

print(weights[0])  # <tf.Variable 'dense/bias:0' shape=(128,) dtype=float32, numpy=array([0., ..., 0.], dtype=float32)>

```