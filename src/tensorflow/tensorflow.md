TensorFlow
==========

TensorFlow uses:

* **Tensor**: multi-dimensional array (numpy ndarray's in Python) with fixed shape and dtype; e.g., batch of image samples [batches, width, height, channels]
* **Nodes**: operators (ops) transforming calculations on tensors, returning new tensors; in deep learning nodes are most likely network layers
* **Graph** of nodes: represents the computation

The flow runs in a session, and all calculations are symbolic so that actually running them runs completely outside of Python (instead of swapping from python to GPU to python for each step).
