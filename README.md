# [Keyword Detection - blog on DLology](https://www.dlology.com/blog/simple-speech-keyword-detecting-with-depthwise-separable-convolutions/)

This repository consists of the tensorflow models and training scripts used 
in the paper: 
[Hello Edge: Keyword spotting on Microcontrollers](https://arxiv.org/pdf/1711.07128.pdf). 
The scripts are adapted from [Tensorflow examples](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands) 
and some are repeated here for the sake of making these scripts self-contained.

To train a DNN with 3 fully-connected layers with 128 neurons in each layer, run:

```
python train.py --model_architecture dnn --model_size_info 128 128 128 
```
The command line argument *--model_size_info* is used to pass the neural network layer
dimensions such as number of layers, convolution filter size/stride as a list to models.py, 
which builds the tensorflow graph based on the provided model architecture 
and layer dimensions. 
For more info on *model_size_info* for each network architecture see 
[models.py](models.py).
The training commands with all the hyperparameters to reproduce the models shown in the 
[paper](https://arxiv.org/pdf/1711.07128.pdf) are given [here](train_commands.txt).

To run inference on the trained model from a checkpoint on train/val/test set, run:
```
python test.py --model_architecture dnn --model_size_info 128 128 128 --checkpoint 
<checkpoint path>
```

To freeze the trained model checkpoint into a .pb file, run:
```
python freeze.py --model_architecture dnn --model_size_info 128 128 128 --checkpoint 
<checkpoint path> --output_file dnn.pb
```

## Pretrained models

Trained models (.pb files) for different neural network architectures such as DNN,
CNN, Basic LSTM, LSTM, GRU, CRNN and DS-CNN shown in 
this [arXiv paper](https://arxiv.org/pdf/1711.07128.pdf) are added in 
[Pretrained_models](Pretrained_models). Accuracy of the models on validation set, 
their memory requirements and operations per inference are also summarized in the 
following table.

<img src="https://user-images.githubusercontent.com/34459978/34018008-0451ef9a-e0dd-11e7-9661-59e4fb4a8347.png">

To run an audio file through the trained model (e.g. a DNN) and get top prediction, 
run:
```
python label_wav.py --wav <audio file> --graph Pretrained_models/DNN/DNN_S.pb 
--labels Pretrained_models/labels.txt --how_many_labels 1
```
```
python label_wav.py --wav ./audio/six_clip_000f49605.wav --graph Pretrained_models/DS_CNN/DS_CNN_S.pb --labels Pretrained_models/labels.txt --how_many_labels 1
```
To explore a pre-trained model structure in TensorBoard
First run this cell then open a TensorBoard in directory `./logs/DS_CNN/DS_CNN_L`
```python
import tensorflow as tf
from tensorflow.python.platform import gfile
with tf.Session() as sess:
    model_filename ='Pretrained_models/DS_CNN/DS_CNN_L.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
LOGDIR='./logs/DS_CNN/DS_CNN_L'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
```

`tensorboard --logdir=./logs/DS_CNN/DS_CNN_L`


## Quantization Guide and Deployment on Microcontrollers

A quick guide on quantizing the KWS neural network models is [here](Deployment/Quant_guide.md). 
The example code for running a DNN model on a Cortex-M development board is also provided [here](Deployment). 

Originally from  [ARM-software](https://github.com/ARM-software)/**[ML-KWS-for-MCU](https://github.com/ARM-software/ML-KWS-for-MCU)**
