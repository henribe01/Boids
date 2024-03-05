from boids import Boid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

WIDTH = 100
HEIGHT = 100
NUM_BOIDS = 10


fig, ax = plt.subplots()
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
boids = []
for _ in range(NUM_BOIDS):
    pos = np.random.rand(2) * [WIDTH, HEIGHT]
    vel = np.random.uniform(-1, 1, 2)
    b = Boid(pos, vel, WIDTH, HEIGHT)
    b.plot(ax)
    boids.append(b)

def update(frame: int) -> None:
    updated_artists = []
    for boid in boids:
        boid.update()
        artist = boid.update_plot()
        updated_artists.append(artist)
    return updated_artists

ani = FuncAnimation(fig, update, blit=True, interval=1000/100, save_count=50)
plt.show()    