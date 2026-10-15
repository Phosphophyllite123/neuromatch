#using boolean indexing

import numpy as np
import matplotlib.pyplot as plt


t_max = 150e-3   # second
dt = 1e-3        # second
tau = 20e-3      # second
el = -60e-3      # milivolt
vr = -70e-3      # milivolt
vth = -50e-3     # milivolt
r = 100e6        # ohm
i_mean = 25e-11  # ampere


#plot_all function
def plot_all(t_range, v_n, spikes=None, spikes_mean=None):
    n = v_n.shape[0]  # 神经元数量
    plt.figure()

    # 子图1：膜电位
    ax1 = plt.subplot(3, 1, 1)
    for j in range(n):
        plt.scatter(t_range, v_n[j], color="k", marker=".", alpha=0.01)
    v_mean = np.mean(v_n, axis=0)
    plt.plot(t_range, v_mean, 'C1', alpha=0.8, linewidth=3)
    plt.ylabel('$V_m$ (V)')
    plt.title('Membrane potential')

    # 子图2：放电光栅图
    plt.subplot(3, 1, 2, sharex=ax1)
    if spikes is not None:
        for j in range(n):
            times = np.array(spikes[j])
            if len(times) > 0:
                plt.scatter(times, j * np.ones_like(times),
                            color="C0", marker=".", alpha=0.2)
    plt.ylabel('neuron')
    plt.title('Spikes')

    # 子图3：平均放电率
    plt.subplot(3, 1, 3, sharex=ax1)
    if spikes_mean is not None:
        plt.plot(t_range, spikes_mean)
    plt.xlabel('time (s)')
    plt.ylabel('rate (Hz)')
    plt.title('Mean spike rate')

    plt.tight_layout()
    plt.show()



# Set random number generator
np.random.seed(2020)

# Initialize step_end, t_range, n, v_n and i
t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max / dt)**(0.5) * (2 * np.random.random([n, step_end]) - 1))

# Initialize spikes and spikes_n
spikes = {j: [] for j in range(n)}
spikes_n = np.zeros([step_end])

# Initialize t_ref and last_spike
t_ref = 0.01
last_spike = -t_ref * np.ones([n])


# Loop over time steps
for step, t in enumerate(t_range):

  # Skip first iteration
  if step == 0:
    continue

  # Compute v_n
  v_n[:, step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r*i[:, step])

  # Initialize boolean numpy array `spiked` with v_n > v_thr
  spiked = (v_n[:, step] > vth)

  # Set relevant values of v_n to resting potential using spiked
  v_n[spiked,step] = vr

  # Collect spike times
  for j in np.where(spiked)[0]:
    spikes[j] += [t]
    spikes_n[step] += 1

# Collect mean spiking rate
spikes_mean = spikes_n / n

# Initialize boolean numpy array clamped using last_spike, t and t_ref
clamped = (last_spike + t_ref > t)

# Reset clamped neurons to vr using clamped
v_n[clamped,step] = vr

# Update numpy array last_spike with time t for spiking neurons
last_spike[spiked] = t

# Plot multiple realizations of Vm, spikes and mean spike rate
plot_all(t_range, v_n, spikes=spikes, spikes_mean=spikes_mean)