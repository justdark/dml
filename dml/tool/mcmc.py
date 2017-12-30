from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math
PI = 3.14159265358979323846
def domain_random():
    return np.random.random()*3.8-1.9
def get_p(x):
    # return 1/(2*PI)*np.exp(- x[0]**2 - x[1]**2)
    return 1/(2*PI*math.sqrt(1-0.25))*np.exp(-1/(2*(1-0.25))*(x[0]**2 -x[0]*x[1] + x[1]**2))

def get_tilde_p(x):
    return get_p(x)*20

def partialSampler(x,dim):
    xes = []
    for t in range(10):
        xes.append(domain_random())
    tilde_ps = []
    for t in range(10):
        tmpx = x[:]
        tmpx[dim] = xes[t]
        tilde_ps.append(get_tilde_p(tmpx))

    norm_tilde_ps = np.asarray(tilde_ps)/sum(tilde_ps)
    u = np.random.random()
    sums = 0.0
    for t in range(10):
        sums += norm_tilde_ps[t]
        if sums>=u:
            return xes[t]


def plotContour(plot = False):
    X = np.arange(-2, 2, 0.05)
    Y = np.arange(-2, 2, 0.05)
    X, Y = np.meshgrid(X, Y)
    Z = get_p([X,Y])
    plt.contourf(X, Y, Z, 100, alpha = 1.0, cmap =cm.coolwarm)

    # plt.contour(X, Y, Z, 7, colors = 'black', linewidth = 0.01)
    if plot:
        plt.show()

def plot3D():
    X = np.arange(-2, 2, 0.05)
    Y = np.arange(-2, 2, 0.05)
    X, Y = np.meshgrid(X, Y)
    Z = get_p([X,Y])
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.grid(False)
    ax.w_yaxis.set_pane_color((1,1,1,0))
    ax.w_xaxis.set_pane_color((1,1,1,1))
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
    plt.show()

# plotContour()
# plot3D()

def metropolis(x):
    new_x = (domain_random(),domain_random())
    acc = min(1,get_tilde_p((new_x[0],new_x[1]))/get_tilde_p((x[0],x[1])))
    u = np.random.random()
    if u<acc:
        return new_x
    return x

def testMetropolis(counts = 100,drawPath = False):
    plotContour()
    x = (domain_random(),domain_random())
    xs = [x]

    for i in range(counts):
        xs.append(x)
        x = metropolis(x)
    if drawPath:
        plt.plot(map(lambda x:x[0],xs),map(lambda x:x[1],xs),'k-',linewidth=0.5)

    plt.scatter(map(lambda x:x[0],xs),map(lambda x:x[1],xs),c = 'g',marker='.')
    plt.show()
    pass

def gibbs(x):
    rst = np.asarray(x)[:]
    path = [(x[0],x[1])]
    for dim in range(2):
        new_value = partialSampler(rst,dim)
        rst[dim] = new_value
        path.append([rst[0],rst[1]])
    return rst,path

def testGibbs(counts = 100,drawPath = False):
    plotContour()

    x = (domain_random(),domain_random())
    xs = [x]
    paths = [x]
    for i in range(counts):
        xs.append([x[0],x[1]])
        x,path = gibbs(x)
        paths.extend(path)
    if drawPath:
        plt.plot(map(lambda x:x[0],paths),map(lambda x:x[1],paths),'k-',linewidth=0.5)
    plt.scatter(map(lambda x:x[0],xs),map(lambda x:x[1],xs),c = 'g',marker='.')
    plt.show()
    pass

testMetropolis(5000,False)
# testGibbs(5000,False)
