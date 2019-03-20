# SNU Chainer
Spiking neural unit implemented with Chainer-v5  
See: https://arxiv.org/abs/1812.07040

## Requirements
- Chainer v5
- cupy
- tqdm

## Usage
### Train the network
Run, `python train.py -g 0 -b 128 -e 100 -t 10`  
(gpu, batch, epoch, simulation time steps)   
<img src="https://github.com/takyamamoto/SNU_Chainer/blob/master/imgs/loss_acc.png" width=60%>

### Test output activity
Run, `python analysis.py -g 0 -m ./results/model`  
<img src="https://github.com/takyamamoto/SNU_Chainer/blob/master/imgs/results.png" width=60%>

### Check SNU layer
Run, `python check_snu_layer.py`  
<img src="https://github.com/takyamamoto/SNU_Chainer/blob/master/imgs/Check_SNU_result.png" width=50%>

### Check Jittered MNIST
Run, `check_jittered_mnist.py`  
<img src="https://github.com/takyamamoto/SNU_Chainer/blob/master/imgs/JitteredMNIST.png" width=50%>
