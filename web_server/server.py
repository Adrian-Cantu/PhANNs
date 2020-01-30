#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import time
import zmq
import Phanns_f
import ann_config

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

while True:
    #  Wait for next request from client
    filename = socket.recv()
    #print("Received request: %s" % message)
    test=Phanns_f.ann_result('uploads/'+filename.decode('UTF-8'))
    (names,pp)=test.predict_score()
    #  Do some 'work'
    #time.sleep(1)

    #  Send reply back to client
    socket.send(b"World")
