import matplotlib.pyplot as plt

folder = 'MOT20-01'
f = open(folder+".txt", "r")
data = eval(f.read())
f.close()
plt.figure()
plt.plot(data)
plt.show()
plt.savefig(folder+'fig.png')
# data = data.strip('[')
# data = data.strip(']')
# data = data.split('.')