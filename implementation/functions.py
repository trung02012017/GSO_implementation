import numpy as np
import os

def f1(particle):
    return sum([particle[i]**2 for i in range(particle.shape[0])])


def f2(particle):
    x = np.abs(particle)
    return np.sum(x) + np.prod(x)


def f3(particle):
    fitness = 0
    for i in range(particle.shape[0]):
        for j in range(i+1):
          fitness += particle[j]
    return fitness


def f4(particle):
    x = np.abs(particle)
    return np.max(x)


def f5(particle):
    return sum([(100*((particle[i+1] - particle[i]**2)**2) + (particle[i] -1)**2) for i in range(particle.shape[0]-1)])


def f6(particle):
    return np.sum([(particle[i] + 0.5)**2 for i in range(particle.shape[0])])


def f7(particle):
    return np.sum([(i+1)*particle[i]**4 for i in range(particle.shape[0])]) + np.random.rand()


def f8(particle):
    return particle[0]**2 + 10e6*np.sum([particle[i]**6 for i in range(1, particle.shape[0])])


def f9(particle):
    return 10e6*particle[0]**2 + np.sum([particle[i]**6 for i in range(1, particle.shape[0])])


def f10(particle):
    return (particle[0] - 1)**2 + np.sum([i*(2*particle[i]**2 - particle[i-1])**2 for i in range(1, particle.shape[0])])


def f11(particle):
    return np.sum([((10e6)**((i)/(particle.shape[0]-1)))*particle[i]**2 for i in range(2, particle.shape[0])])


def f12(particle):
    return np.sum([(i+1)*particle[i]**2 for i in range(particle.shape[0])])


def f13(particle):
    return np.sum([particle[i]**2 for i in range(particle.shape[0])]) + \
           (np.sum([0.5*i*particle[i]**2 for i in range(particle.shape[0])]))**2 + \
           (np.sum([particle[i]**2 for i in range(particle.shape[0])]))**4


def f14(particle):
    return np.sum([-1*particle[i]*np.sin(np.abs(particle[i])**0.5) for i in range(particle.shape[0])])


def f15(particle):
    return np.sum([particle[i]**2 - 10*np.cos(2*np.pi*particle[i]) + 10 for i in range(particle.shape[0])])


def f16(particle):
    return -20*np.exp(-0.2*(1/particle.shape[0]*np.sum([particle[i]**2 for i in range(particle.shape[0])]))**0.5) - \
           np.exp(1/particle.shape[0]*np.sum([np.cos(2*np.pi*particle[i]) for i in range(particle.shape[0])])) + 20 + \
           np.e

def f17(particle):
    return 1/4000*np.sum([particle[i]**2 for i in range(particle.shape[0])]) - \
           np.prod([np.cos(particle[i]/((i+1)**0.5)) for i in range(particle.shape[0])]) + 1


def f18(particle):
    return 0


def f19(particle):
    return 0


def f20(particle):
    return np.sum([((np.sum([(0.5**k)*np.cos(2*np.pi*(3**k)*(particle[i]+0.5)) for k in range(21)])) -
                   (particle.shape[0]*np.sum([(0.5**j)*np.cos(np.pi*(3**j)) for j in range(21)])))
                   for i in range(particle.shape[0])])


def f21(particle):
    return np.sum([np.abs(particle[i]*np.sin(particle[i]) + 0.1*particle[i]) for i in range(particle.shape[0])])


def f22(particle):
    return 0.5 + ((np.sin(np.sum([particle[i]**2 for i in range(particle.shape[0])])))**2 - 0.5)*\
           (1+0.001*(np.sum([particle[i]**2 for i in range(particle.shape[0])])))**-2


def f23(particle):
    return 1/particle.shape[0]* \
           np.sum([particle[i]**4 - 16*particle[i]**2 + 5*particle[i] for i in range(particle.shape[0])])


def f24(particle):
    return np.sum([particle[i]**2 + 2*particle[i+1]**2 - 0.3*np.cos(3*np.pi*particle[i]) -
                   0.4*np.cos(4*np.pi*particle[i+1]) + 0.7 for i in range(particle.shape[0]-1)])


def f25(particle):
    return -1*(-0.1*np.sum([np.cos(5*np.pi*particle[i]) for i in range(particle.shape[0])]) -
               np.sum([particle[i]**2 for i in range(particle.shape[0])]))


# a = np.array([1, 2, 3, 4])
# b = np.array([0, 0, 0, 0])
# print(f25(a))
# print(f25(b))

path = 'results/f'
models = ['GA', 'PSO', 'Bee', 'MWOA', 'GSO1(PSO+PSO)', 'GSO2(PSO+Bee)', 'GSO3(PSO+GA)', 'IGSO(GSO+MWOA)']
for i in range(1, 26):
    for model in models:
        real_path = path + str(i) + '/'
        real_path += str(model)
        os.mkdir(real_path)

