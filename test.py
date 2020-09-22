import numpy as np
from scipy import signal

results = np.random.randint(1,4,(10,2))
# print(x)
# y = signal.resample(x,5,axis=1)
Results = {'without retraining':results[:,0],'retrain and validate on 4 subject splits':results[:,1]}
print(1)