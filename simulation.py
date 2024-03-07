from boids import Boid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

WIDTH = 50
HEIGHT = 50
NUM_BOIDS = 20


class Flock:
    def __init__(self, num_boids: int, width: int, height: int, radius: float) -> None:
        # Initialize the boids
        self.boids = []
        for _ in range(num_boids):
            pos = np.random.rand(2) * [width, height]
            vel = np.random.uniform(-1, 1, 2)
            b = Boid(pos, vel, width, height)
            self.boids.append(b)

        self.radius = radius

        # Create the plot
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.artists = [b.plot(self.ax) for b in self.boids]
        self.ani = FuncAnimation(self.fig, self.update, blit=True, interval=1000/60, cache_frame_data=False)
    def show(self) -> None:
        plt.show()

    def update(self, frame: int) -> list:
        updated_artists = []
        for boid in self.boids:
            neighborhood = boid.get_neighborhood(self.boids, self.radius)
            boid.update(neighborhood)
            artist = boid.update_plot()
            updated_artists.append(artist)
        return updated_artists


if __name__ == "__main__":
    f = Flock(NUM_BOIDS, WIDTH, HEIGHT, 10)
    f.show()
