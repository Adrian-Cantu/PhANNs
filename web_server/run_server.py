import time
import Phanns_f
import ann_config
import os


done=dict()

for mydir in ['saves','uploads','csv_saves']:
    files=os.listdir(mydir)
    for f in files:
        if not f.startswith('.'):
            done[f]=True
            #os.remove(os.path.join(mydir, f))

done=dict()

while True:
    #  Wait for next request from client
 #   filename = socket.recv()
 #   print("Received request: uploads/"+filename.decode('UTF-8'))
 #   test=Phanns_f.ann_result('uploads/'+filename.decode('UTF-8'))
 #   (names,pp)=test.predict_score()
    #  Do some 'work'
    
    
    
    time.sleep(1)
    files=os.listdir('uploads')
    for f in files:
        if not f.startswith('.'):
            if f not in done.keys():
                print(f + ' is a new file')
                test=Phanns_f.ann_result('uploads/'+f)
                (names,pp)=test.predict_score()
                done[f]=True

    #  Send reply back to client
    #socket.send(b"World")
