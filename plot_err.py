import matplotlib.pyplot as plt
import numpy as np

training_err = np.array([0.0323, 0.0378, 0.0506, 0.0702])
test_err = np.array([0.0295, 0.0385, 0.0567, 0.0888])

plt.loglog(training_err, test_err, 'o-')
for i in range(4):
	plt.annotate(str(100/(2**i))+'%'+'data', (training_err[i], test_err[i]))
plt.xlabel("training error")
plt.ylabel("test error")
plt.show()