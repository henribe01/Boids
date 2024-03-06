import numpy as np
import matplotlib.pyplot as plt

class Boid:
    def __init__(self, pos: np.ndarray, vel: np.ndarray, WIDTH: int, HEIGHT: int):
        self.pos = pos
        self.vel = vel
        self.artist = None
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT

        self.cohesion_weight = 1
        self.alignment_weight = 1
        self.separation_weight = 1

    def plot(self, ax: plt.Axes) -> plt.Artist:
        """
        Plots the boid as an triangle pointing in the direction of the velocity on the given axis.
        """
        # TODO: For now, I just plot the position of the boid as a point. Replace this with a triangle.
        self.artist, = ax.plot(self.pos[0], self.pos[1], 'ro', markersize=3)

    def update_plot(self) -> plt.Artist:
        """
        Updates the position of the boid's plot and returns the artist object.
        """
        assert self.artist is not None, "plot() must be called before update_plot()"
        self.artist.set_data(self.pos[0], self.pos[1])
        return self.artist

    def update(self, neighborhood: list, max_speed: float = 4, max_force: float = 1) -> None:
        """
        Updates the position and velocity of the boid based on its current acceleration.
        """
        self.check_bounds(self.WIDTH, self.HEIGHT)

        alignment = self.align(neighborhood)
        if np.linalg.norm(alignment) > max_force:
            alignment = max_force * alignment / np.linalg.norm(alignment)
        
        cohesion = self.cohesion(neighborhood)
        if np.linalg.norm(cohesion) > max_force:
            cohesion = max_force * cohesion / np.linalg.norm(cohesion)

        separation = self.separation(neighborhood)
        if np.linalg.norm(separation) > max_force:
            separation = max_force * separation / np.linalg.norm(separation)

        # Update the velocity
        acc = alignment * self.alignment_weight + cohesion * self.cohesion_weight + separation * self.separation_weight
        self.vel += acc
        if np.linalg.norm(self.vel) > max_speed:
            self.vel = max_speed * self.vel / np.linalg.norm(self.vel)
        self.pos += self.vel


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

    def get_neighborhood(self, boids: list, radius: float) -> list:
        """
        Returns a list of boids within the given radius of this boid
        """
        # TODO: Add angle to the condition
        neighborhood = []
        for boid in boids:
            width_vec = [self.WIDTH, 0]
            height_vec = [0, self.HEIGHT]
            distances = [
                np.linalg.norm(boid.pos - self.pos),
                np.linalg.norm(boid.pos - (self.pos + width_vec)),
                np.linalg.norm(boid.pos - (self.pos + height_vec)),
                np.linalg.norm(boid.pos - (self.pos - width_vec)),
                np.linalg.norm(boid.pos - (self.pos - height_vec))
            ]
            if min(distances) < radius and boid != self:
                neighborhood.append(boid)
        return neighborhood
    
    def align(self, neighborhood: list) -> np.ndarray:
        """
        Returns the alignment vector of the boid based on the velocities of its neighbors.
        """
        if len(neighborhood) == 0:
            return np.zeros(2)
        avg_vel = np.mean([boid.vel for boid in neighborhood], axis=0)
        return avg_vel - self.vel

    def cohesion(self, neighborhood: list) -> np.ndarray:
        """
        Returns the cohesion vector of the boid based on the positions of its neighbors.
        """
        if len(neighborhood) == 0:
            return np.zeros(2)
        avg_pos = np.mean([boid.pos for boid in neighborhood], axis=0)
        #TODO: Set magnitude to max_speed
        return avg_pos - self.pos - self.vel
    
    def separation(self, neighborhood: list) -> np.ndarray:
        """
        Returns the separation vector of the boid based on the positions of its neighbors.
        """
        if len(neighborhood) == 0:
            return np.zeros(2)
        separation = np.zeros(2)
        for boid in neighborhood:
            diff = self.pos - boid.pos
            separation += diff / np.linalg.norm(diff)*
        return separation / len(neighborhood) - self.vel
