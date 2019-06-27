# ANN_site

##to add the capacity of binding low ports
sudo setcap 'cap_net_bind_service=+ep' /home/ubuntu/anaconda3/envs/tensorflow_p36/bin/python3.6
gunicorn --workers 3 --bind 0.0.0.0:80 -m 007 wsgi:app



#set and run gunicorn
sudo cp ANN_site.service /etc/systemd/system/
sudo systemctl start ANN_site

#set nginx 
sudo cp nginx_ANN_site /etc/nginx/sites-enabled/ANN_site
sudo rm /etc/nginx/sites-enabled/default
sudo systemctl restart nginx


#st firewall rules

sudo ufw allow 'Nginx Full'

#start rq

rq worker microblog-tasks
