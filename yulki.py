import time
import datetime

start_time = time.time()
start_datetime = datetime.datetime.now()


end_datetime = datetime.datetime.now()
end_time = time.time()

print(f'start time :{start_datetime} \nend time :{end_datetime} \ntotal time :{end_time - start_time}')


st = '  sss_  '
print(st.replace('_', ' ').strip())


a = [0]
a += [1, 2, 3]
print(a)

import torch
a = torch.ones(100,100).cuda()
print(a)