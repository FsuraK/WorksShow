import torch
import numpy as np
import matplotlib.pyplot as plt
import math

def plotimage():
    plt.close()

    x = np.arange(0, len(true), 1)
    plt.figure("name")
    plt.plot(x, np.array(true),"r",x,np.array(apm),"g--")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("result")
    plt.show()

def function_true(x_next):
    y_next_true = math.exp(0.5*x_next**2)
    return y_next_true

def function_apm(x_now,x_next,y_now):
    y_next_apm = y_now+x_now*y_now*(x_next-x_now)
    y_next_apm = y_now + 0.5*(x_now * y_now+x_next*y_next_apm) * (x_next - x_now)
    return y_next_apm

def function_true_sin(x_next):
    y_next_true = math.exp(math.sin(x_next))
    return y_next_true

def function_apm_sin(x_now,x_next,y_now):
    y_next_apm = y_now+math.cos(x_now)*y_now*(x_next-x_now)
    y_next_apm = y_now + 0.5*( math.cos(x_now)*y_now + math.cos(x_next)*y_next_apm ) * (x_next - x_now)
    return y_next_apm

x_now = 0
y_now = 1
y_next_apm = 1
true = []
apm = []
for i in range(3000):
    x_now = x_now+0.01
    x_next = x_now+0.01
    y_now = y_next_apm

    y_next_apm = function_apm_sin(x_now,x_next,y_now)
    apm.append(y_next_apm)
    y_next_true = function_true_sin(x_next)
    true.append(y_next_true)

plotimage()
