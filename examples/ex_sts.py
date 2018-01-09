from SampleMethods import *
from RunModel import RunModel
from module_ import def_model, def_target, readfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

filename = 'input_sts.txt'
data = readfile(filename)

# extract input data
model = def_model(data['Model'])
method = data['Method']
N = data['Number of Samples']
d = data['Stochastic dimension']
pdf = data['Probability distribution (pdf)']
pdf_params = data['Probability distribution parameters']
sts_design= data['STS design']


current_dir = os.getcwd()
path = os.path.join(os.sep, current_dir, 'results')
os.makedirs(path, exist_ok=True)
os.chdir(path)


'''
Initialize 
1. class Sampling methods
2. class RunModel
'''

sm = SampleMethods(distribution=pdf, dimension=d, parameters=pdf_params, method=method)
rm = RunModel(model=model)


'''
Run code
'''

sts = sm.STS(sm, strata=Strata(nstrata=sts_design))
fx = rm.Evaluate(rm, sts.xi)

'''
Plots
'''

subpath = os.path.join(os.sep, path, 'sts')
os.makedirs(subpath, exist_ok=True)
os.chdir(subpath)

np.savetxt('samples.txt', sts.xi, delimiter=' ')
np.savetxt('model.txt', fx.v)

plt.figure()
plt.scatter(sts.xi[:, 0], sts.xi[:, 1])
plt.savefig('samples.png')

plt.figure()
n, bins, patches = plt.hist(fx.v, 50, normed=1, facecolor='g', alpha=0.75)
plt.title('Histogram')
plt.savefig('histogram.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sts.xi[:, 0], sts.xi[:, 1], fx.v, c='r', s=2)
plt.gca().invert_xaxis()
plt.savefig('model.png')


os.chdir(current_dir)