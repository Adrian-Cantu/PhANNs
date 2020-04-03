import zmq
from zmq.devices.basedevice import ProcessDevice
from multiprocessing import Process
import random
import time

frontend_port = 5559
backend_port = 5555
number_of_workers = 10



queuedevice = ProcessDevice(zmq.QUEUE, zmq.XREP, zmq.XREQ)
queuedevice.bind_in("tcp://127.0.0.1:%d" % frontend_port)
queuedevice.bind_out("tcp://127.0.0.1:%d" % backend_port)
queuedevice.setsockopt_in(zmq.RCVHWM, number_of_workers)
queuedevice.setsockopt_out(zmq.SNDHWM, number_of_workers)
queuedevice.start()
time.sleep (2)  

while True:
	pass
