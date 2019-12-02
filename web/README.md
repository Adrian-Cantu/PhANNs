# ANN_site
This yet to be named tool uses an artificial neural network to classify phage structural proteins. It was trained on a manually curated database of more than 200,000 phage proteins. It can predict the fallowing structural classes:

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

Clone this repository, download the [model file](https://edwards.sdsu.edu/phanns/tri_p.h5) and put  it in the tri_p_model directory. 

You will need a Redis server installed. Other than that, all requirement are listed on the `environment.yml` file, but the easier way to install them is using anaconda

```
conda env create -f environment.yml
```

then activate the environment

```
conda activate tf_gpu
```

start a rq worker (probably a good idea to do it in a separate terminal)

```
rq worker microblog-tasks &
```
and finally start the webserver

```
gunicorn --workers 3 --timeout 20000000  --bind 0.0.0.0:8080 -m 007 wsgi:app
```

You should be able to see the server running at < 0.0.0.0:8080>.
