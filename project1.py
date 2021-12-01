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

def find_characteristics(ts, zs, ws, printout=True):
    w_max = np.amax(ws) # max vertical velocity of parcel
    z_max = np.amax(zs) # max altitude of parcel
    z_w_max = zs[np.argmax(ws)] # altitude where max velocity is reached

    # finding the period
    t_zero_crossing = ts[np.where(zs<1e-3)]
    if len(t_zero_crossing)>1:
        period = t_zero_crossing[1]-t_zero_crossing[0]
        N = 2*np.pi/period
    else:
        period = "Could not be found"
        N = "Could not be found"

    if printout:
        print("Max altitude: ", z_max, "m")
        print("Max velocity: ", w_max, "m/s")
        print("Max velocity reached at: ", z_w_max, "m altitude")
        print("Period: ", period, "s")
        print("Angular frequency: ", N, "s^{-1}")

    return w_max, z_max, z_w_max, period, N

t, z, w = compute(1000)
t_, z_, w_ = compute(1000, s=False)

w_max, z_max, z_w_max, period, N = find_characteristics(t, z, w)
w_max, z_max, z_w_max, period, N = find_characteristics(t_, z_, w_)


fig1 = plt.figure()
ax1 = fig1.add_subplot(211)
ax2 = fig1.add_subplot(212)

ax1.plot(t, z)
ax1.set_ylabel("Height [m]")

ax2.plot(t, w)
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Vertical Velocity [m/s]")

fig2 = plt.figure()
ax3 = fig2.add_subplot(211)
ax4 = fig2.add_subplot(212)

ax3.plot(t_, z_)
ax3.set_ylabel("Height [m]")

ax4.plot(t_, w_)
ax4.set_xlabel("Time [s]")
ax4.set_ylabel("Vertical Velocity [m/s]")

plt.show()
