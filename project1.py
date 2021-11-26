import numpy as np
import matplotlib.pyplot as plt

# Constants
theta_at_0 = 300
g = 9.81

def get_theta_z(z, theta_at_0=300, simple=True):
    if simple:
        lapse = 0.005
    else:
        if z<=50:
            lapse = -0.04
        elif z<2000:
            lapse = 0
        elif z<=2250:
            lapse = 0.04
        else:
            lapse = 0.005

    theta_0 = theta_at_0 + lapse * z

    return theta_0

def compute(t_tot, dt=0.1, w_0=0, theta_star=305, s=True):
    t = np.arange(0, t_tot, dt)
    w = np.zeros(len(t))
    z = np.zeros(len(t))

    w[0] = w_0

    for i in range(len(t)-1):
        w[i+1] = w[i] + g * (theta_star/get_theta_z(z[i], simple=s) - 1) * dt

        z[i+1] = z[i] + w[i+1] * dt

    return t, z, w

t, z, w = compute(1000)
t_, z_, w_ = compute(1000, s=False)

plt.figure(1)
plt.plot(t, z)
plt.xlabel("Time [s]")
plt.ylabel("Height [m]")

plt.figure(2)
plt.plot(t_, z_)
plt.xlabel("Time [s]")
plt.ylabel("Height [m]")

plt.show()
