import pennylane as qml
from pennylane import numpy as np

dev = qml.device("forest.qvm", device="2q", noisy=True)
 
@qml.qnode(dev)
def circuit(parmas):
    qml.RX(parmas[0], wires=0)
    qml.RY(parmas[1], wires=1)
    qml.CNOT(wires=[0,1])
    return qml.probs(wires=[0,1])


def cost(x):
    p = circuit(x)
    return p[0]**2 + (0.5-p[1])**2 + (0.5-p[2])**2 + p[3]**2


def GDO(steps, cost):
    print(f'doing measurement for {steps} times')
    params = np.array([0.0000001,0.0000001])
    cost(params)
    opt = qml.GradientDescentOptimizer(stepsize=0.5)
    for i in range(steps):
        params = opt.step(cost, params)
        #print('Cost after step {:5d}: {: .7f}'.format(i+1, cost(params)))
    #print('Optimized rotation angle: {}'.format(params),'\n')
    print('probability of [00,01,10,11] is :')
    print(circuit(params), '\n')

GDO(1,cost)
GDO(10,cost)
GDO(100, cost)
GDO(1000,cost) 