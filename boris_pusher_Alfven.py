import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

E0 = 0.2       
c  = 1.0      
k  = 0.5       
omega = c * k 
phi0 = 0.0


plt.ion()

class Particle: 
    def __init__(self, q, m, r, v):
        self.q = q  # charge
        self.m = m  # mass
        self.r = np.array(r, dtype=float)  # position vector
        self.v = np.array(v, dtype=float)  # velocity vector

    def __repr__(self):
        return f"Particle(q={self.q}, m={self.m}, r={self.r}, v={self.v})"
    
    def boris_pusher(self, E, B, dt):
        #Half acceleration from the E field 
        v_minus = self.v + (self.q * dt / (2 * self.m)) * E
        #Rotation due to the B field
        t_vec = (self.q * dt / (2 * self.m)) * B
        t_mag_sq = np.dot(t_vec, t_vec)
        s_vec = 2 * t_vec / (1 + t_mag_sq)

        #First rotation 
        v_prime = v_minus + np.cross(v_minus, t_vec)
        v_plus = v_minus + np.cross(v_prime, s_vec)

        #Second half acceleration from the E field
        v_n_plus_half = v_plus + (self.q * dt / (2 * self.m)) * E
        #update particle 
        self.v = v_n_plus_half
        self.r = self.r + self.v * dt
        return self

def alfven_fields(r, t, *, B0, vA, waves, s_elsasser=+1.0):
    z = r[2]
    zhat = np.array([0.0, 0.0, 1.0])

    deltaB = np.zeros(3)

    for w in waves:
        dB   = w["dB"]
        k    = w["k"]
        phi0 = w.get("phi0", 0.0)
        omega = vA * k
        phi = k * z - omega * t + phi0
        # Left-hand circular polarization in x-y 
        deltaB += dB * np.array([np.cos(phi), -np.sin(phi), 0.0])

    B = np.array([0.0, 0.0, B0]) + deltaB
    E = s_elsasser * vA * np.cross(zhat, deltaB)

    return E, B

if __name__ == "__main__":
    dt = 0.01
    num_steps = 4000
    q = +1.0
    m = 1836.0  
    B0 = 1.0
    vA = 0.2   
    # Three wavelengths
    waves = [
        {"dB": 0.05 * B0, "k": 0.5, "phi0": 0.0},
        {"dB": 0.05 * B0, "k": 1.0, "phi0": 1.2},
        {"dB": 0.05 * B0, "k": 2.0, "phi0": 2.0},
    ]
    initial_r = [0.0, 0.0, 0.0]
    v_perp = 0.3
    v_par  = 0.1
    initial_v = [v_perp, 0.0, v_par]

    proton = Particle(q=q, m=m, r=initial_r, v=initial_v)

    print ("Initial State:", proton)

    r_history = [proton.r.copy()]

    for i in range(num_steps):
        t = i * dt
        t_half = (i + 0.5) * dt
        r_half = proton.r + 0.5 * proton.v * dt

        E_field, B_field = alfven_fields(
            r_half, t_half,
            B0=B0, vA=vA, waves=waves,
            s_elsasser=+1.0
        )

        proton = proton.boris_pusher(E_field, B_field, dt)
        r_history.append(proton.r.copy())

    r_history = np.array(r_history)
    x, y, z = r_history[:, 0], r_history[:, 1], r_history[:, 2]

    print(f"Final position: ({x[-1]}, {y[-1]}, {z[-1]})")
    print (f"Final velocity: {proton.v}")

    #plotting the trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, marker=".", linewidth=1)

    # Mark starting and ending points
    ax.scatter(x[0], y[0], z[0], color="green", marker="o", s=100)  # start
    ax.scatter(x[-1], y[-1], z[-1], color="red", marker="o", s=100)  # end

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Electron trajectory in 3D (B along +z)")
    ax.view_init(elev=30, azim=45)  
    plt.show()

    # Animation of the trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    line, = ax.plot([], [], [], lw=1.5)
    point, = ax.plot([], [], [], "ro")

    margin = 0.2
    ax.set_xlim(x.min() - margin, x.max() + margin)
    ax.set_ylim(y.min() - margin, y.max() + margin)
    ax.set_zlim(z.min() - margin, z.max() + margin)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Boris electron trajectory (3D Animation)")
    ax.view_init(elev=30, azim=45)

    for i in range(len(x)):
        line.set_data(x[:i], y[:i])
        line.set_3d_properties(z[:i])
        point.set_data(x[i:i+1], y[i:i+1])
        point.set_3d_properties(z[i:i+1])
        plt.pause(0.01)

    plt.ioff()
    plt.show()