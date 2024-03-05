import numpy as np
import matplotlib.pyplot as plt

class Boid:
    def __init__(self, pos: np.ndarray, vel: np.ndarray, WIDTH: int, HEIGHT: int):
        self.pos = pos
        self.vel = vel
        self.acc = np.zeros(2)
        self.artist = None
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT

    def plot(self, ax: plt.Axes) -> plt.Artist:
        """
        Plots the boid as an triangle pointing in the direction of the velocity on the given axis.
        """
        # TODO: For now, I just plot the position of the boid as a point. Replace this with a triangle.
        self.artist, = ax.plot(self.pos[0], self.pos[1], 'ro')

    def update_plot(self) -> plt.Artist:
        """
        Updates the position of the boid's plot and returns the artist object.
        """
        assert self.artist is not None, "plot() must be called before update_plot()"
        self.artist.set_data(self.pos[0], self.pos[1])
        return self.artist

    def update(self) -> None:
        """
        Updates the position and velocity of the boid based on its current acceleration.
        """
        self.pos += self.vel
        self.check_bounds(self.WIDTH, self.HEIGHT)
        self.vel += self.acc
        self.acc = np.zeros(2)

    def check_bounds(self, width: int, height: int) -> None:
        """
        If the boid goes out of bounds, it wraps around to the other side.
        """
        if self.pos[0] < 0:
            self.pos[0] = width
        elif self.pos[0] > width:
            self.pos[0] = 0
        if self.pos[1] < 0:
            self.pos[1] = height
        elif self.pos[1] > height:
            self.pos[1] = 0

    def _cohesion(self, boids: list) -> np.ndarray:
        """
        Returns the average position of the boids in the given list and returns the vector pointing towards it.
        """
        center = np.zeros(2)
        for boid in boids:
            if boid != self:
                center += boid.pos
        return center / (len(boids) - 1) - self.pos