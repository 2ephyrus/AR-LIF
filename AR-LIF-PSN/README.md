## AR-LIF-PSN

This is the technical documentation for implementing AR-LIF in the [PSN](https://github.com/fangwei123456/Parallel-Spiking-Neuron) code. This code is only for the cifar10dvs dataset. If you need to perform more validation operations on AR-LIF based on PSN, you only need to port the `aLIFSpike` class in `./models.py`.

SpikingJelly is required to run these codes. The version of SpikingJelly should be `>=0.0.0.0.14 ` unless otherwise specified.

Install the SpikingJelly with the version `0.0.0.0.14`:

```
pip install spikingjelly==0.0.0.0.14
```

For usage of codes, refer to the `readme` in `./cifar10dvs` directory.

## Training
    # Before you do this, make sure you have divided the dataset into /train and /test according to the instructions in ./cifar10dvs/readme.md.
    python ./train_vgg.py -b 16 --epochs 100 -TET -T 16 --lamb 0.3