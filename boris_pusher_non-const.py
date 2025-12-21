import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

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

def B_mirror(r, B0=1.0, alpha=0.02):
    x, y, z = r
    Bz = B0 * (1.0 + alpha * z**2)
    dBz_dz = 2 * alpha * B0 * z

    Bx = -0.5 * x * dBz_dz
    By = -0.5 * y * dBz_dz

    return np.array([Bx, By, Bz])


if __name__ == "__main__":
    dt = 0.1     
    q = -1.0  
    m = 1.0           

    initial_r = [0.0, 0.0, 0.0]
    initial_v = [0.5, 0.5, 1.0] 

    # Create the particle
    electron = Particle(q=q, m=m, r=initial_r, v=initial_v)

    print("Initial State")
    print(electron)
    print("---------------------\n")
    num_steps = 200
    r_history = [electron.r.copy()]
    E_field = np.array([0.0, 0.0, 0.0])

    for i in range(num_steps):  
        B_field = B_mirror(electron.r)
        electron = electron.boris_pusher(E_field, B_field, dt)
        r_history.append(electron.r.copy())

    print("\n--- Final State ---")
    print(electron)

    r_history = np.array(r_history)
    x = r_history[:, 0]
    y = r_history[:, 1]
    z = r_history[:, 2]

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