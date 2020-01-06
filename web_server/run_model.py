import Phanns_f
import ann_config
import sys

test=Phanns_f.ann_result(sys.argv[1])
(names,pp)=test.predict_score_single_run()
print(pp)
