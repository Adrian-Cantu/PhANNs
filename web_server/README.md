# PhANNs
PhANNs uses an artificial neural networks to classify phage structural proteins. It was trained on a manually curated database of more than 80,000 phage proteins. It can predict the fallowing structural classes:

1. Major capsid
1. Minor capsid
1. Baseplate
1. Major tail
1. Minor tail
1. Portal
1. Tail fiber
1. Tail shaft
1. Collar
1. Head-Tail Joining
1. Other

## Install?

You don’t have to!! you can use our web server [here](https://edwards.sdsu.edu/phanns)

## But I really want to install it!

Clone this repository, download the [model file](https://edwards.sdsu.edu/phanns/download/model.tar), put it in the "deca\_model" directory and uncompress it. 

All requirement are listed on the `cpu_environment.yml` file, but the easier way to install them is using anaconda

```
conda env create -f cpu_environment.yml
```

then activate the environment

```
conda activate tf2_cpu
```



and finally start the webserver, making sure you use tensorflow as the backend

```
KERAS_BACKEND=tensorflow python ANN_site.py
```

You should be able to see the server running at < 0.0.0.0:8080>.
