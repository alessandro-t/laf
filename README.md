Learning Aggregation Functions - Code
----
This repository contains the python code for reproducing the experiments 
described in the paper [Learning Aggregation Functions]
(https://arxiv.org/abs/2012.08482), by Giovanni Pellegrini,
Alessandro Tibo, Paolo Frasconi, Andrea Passerini, and Manfred Jaeger. 

The current implemention relies on PyTorch. However, a TensorFlow version
of the LAF layer is implemented in `laf/model_tf.py`. 

Python Requirments
----
* `numpy`
* `pytorch (version >= 2.0)`
* `tqdm`
* `torch-scatter` (see instructions 
   on <https://github.com/rusty1s/pytorch_scatter>)
* `torchvision`
	
Run the Experiments for MNIST
----
	$ cd mnist
	$ ./run.sh
	
Citation
----
	@misc{pellegrini2020learning,
      title={Learning Aggregation Functions}, 
      author={Giovanni Pellegrini and Alessandro Tibo and Paolo Frasconi and Andrea Passerini and Manfred Jaeger},
      year={2020},
      eprint={2012.08482},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
    }

