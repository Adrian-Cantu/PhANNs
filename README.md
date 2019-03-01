# ANN_site

##to add the capacity of binding low ports
sudo setcap 'cap_net_bind_service=+ep' /home/ubuntu/anaconda3/envs/tensorflow_p36/bin/python3.6
gunicorn --workers 3 --bind 0.0.0.0:80 -m 007 wsgi:app
