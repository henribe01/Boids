from boids import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

WIDTH = 50
HEIGHT = 50
NUM_BOIDS = 50


class Flock:
    def __init__(self, num_boids: int, width: int, height: int, radius: float) -> None:
        # Initialize the boids
        self.boids = []
        for _ in range(num_boids):
            pos = np.random.rand(2) * [width, height]
            vel = np.random.uniform(-1, 1, 2)
            b = Boid(pos, vel, width, height)
            self.boids.append(b)

        pred = Predator(np.random.rand(2) * [width, height], np.random.uniform(-1, 1, 2), width, height)
        self.boids.append(pred)

        self.radius = radius

        # Create the plot
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.set_aspect("equal")
        self.ani = FuncAnimation(self.fig, self.update, blit=True, interval=1000/30, cache_frame_data=False)

        # Create sliders
        self.fig.subplots_adjust(bottom=0.3)
        axcolor = 'lightgoldenrodyellow'
        sliders = {"Alignment": None,
                     "Cohesion": None,
                     "Separation": None,
                     "Flee": None,
                     "Boundary": None}
        for name in sliders:
            ax = self.fig.add_axes([0.25, 0.02 + 0.05 * list(sliders.keys()).index(name), 0.65, 0.03], facecolor=axcolor)
            sliders[name] = Slider(ax=ax,
                                      label=name,
                                      valmin=0,
                                      valmax=2,
                                      valinit=1,
                                        orientation="horizontal")
            sliders[name].on_changed(self.update_weights)
        self.sliders = sliders

    def update_weights(self, val) -> None:
        for boid in self.boids:
            boid.alignment_weight = self.sliders["Alignment"].val
            boid.cohesion_weight = self.sliders["Cohesion"].val
            boid.separation_weight = self.sliders["Separation"].val
            boid.flee_weight = self.sliders["Flee"].val
            boid.boundary_weight = self.sliders["Boundary"].val


    def show(self) -> None:
        plt.show()

    def update(self, frame: int) -> list:
        updated_artists = []
        for boid in self.boids:
            if isinstance(boid, Predator):
                neighborhood = boid.get_neighborhood(self.boids, 2 * self.radius)
            else:
                neighborhood = boid.get_neighborhood(self.boids, self.radius)
            boid.update(neighborhood)
            artist = boid.update_plot(self.ax)
            updated_artists.append(artist)
        return updated_artists


if __name__ == "__main__":
    f = Flock(NUM_BOIDS, WIDTH, HEIGHT, 10)
    f.show()
