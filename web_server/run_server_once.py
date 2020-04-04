import time
import Phanns_f
import ann_config
import os

files=os.listdir('uploads')
for f in files:
    if not f.startswith('.'):
        print(f + ' is a new file')
        test=Phanns_f.ann_result('uploads/'+f)
        (names,pp)=test.predict_score()
