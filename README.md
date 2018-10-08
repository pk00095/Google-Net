# Google-Net
My attempt at coding GoogleNet with tensorflow using tensorflow layers api


# About
`goog.py` has class __googlenet.py__ which has 3 functions :
* __inception_module__ : inception module as described in the paper
* __inception_module_with_aux_classifier__ : inception module with auxliary classifier as described in the paper
* __build__ :  assembles the network

After the graph is created it written into an event file inside the folder *log* 

For tensorboard vizualization type:

`tensorboard --logdir ./log`
