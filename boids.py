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


    def update_plot(self, ax: plt.Axes, c="b") -> plt.Artist:
        """
        Updates the position of the boid's plot and returns the artist object.
        """
        orientation = np.arctan2(self.vel[1], self.vel[0])
        self.artist, = ax.plot(self.pos[0], self.pos[1], markersize=3, marker=(3, 0, np.degrees(orientation) - 90), linestyle="", color="black")
        return self.artist


    def update(self, neighborhood: list, max_speed: float = 1, max_force: float = 0.1) -> None:
        """
        Updates the position and velocity of the boid based on its current acceleration.
        """
        alignment = self.align(neighborhood)
        cohesion = self.cohesion(neighborhood)
        separation = self.separation(neighborhood)
        bound = self.boundary()

        # Update the velocity
        acc = alignment * self.alignment_weight + cohesion * self.cohesion_weight + separation * self.separation_weight + bound
        if np.linalg.norm(acc) > max_force:
            acc = max_force * acc / np.linalg.norm(acc)
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

    def get_neighborhood(self, boids: list, radius: float) -> dict:
        """
        Returns a dictionary of boids within a certain radius of the current boid and their distances.
        """
        # TODO: Add angle to the condition
        neighborhood = dict()
        pos_array = np.array([boid.pos for boid in boids])
        dist = np.linalg.norm(pos_array - self.pos, axis=1)
        neighborhood = {d: b for d, b in zip(dist, boids) if d < radius and d != 0}
        return neighborhood

    
    def align(self, neighborhood: dict) -> np.ndarray:
        """
        Returns the alignment vector of the boid based on the velocities of its neighbors.
        """
        if len(neighborhood) == 0:
            return np.zeros(2)
        avg_vel = np.mean([boid.vel for boid in neighborhood.values()], axis=0)
        return avg_vel - self.vel

    def cohesion(self, neighborhood: dict) -> np.ndarray:
        """
        Returns the cohesion vector of the boid based on the positions of its neighbors.
        """
        if len(neighborhood) == 0:
            return np.zeros(2)
        avg_pos = np.mean([boid.pos for boid in neighborhood.values()], axis=0)
        return avg_pos - self.pos
    
    def separation(self, neighborhood: dict) -> np.ndarray:
        """
        Returns the separation vector of the boid based on the positions of its neighbors.
        """
        if len(neighborhood) == 0:
            return np.zeros(2)
        pos_diff = np.array([boid.pos - self.pos for boid in neighborhood.values()])
        dist_squared = np.array([dist**2 for dist in neighborhood.keys()])
        separation = -np.sum(pos_diff / dist_squared[:, None], axis=0)
        return separation
    
    def boundary(self) -> np.ndarray:
        """
        Returns a vector pointing away from the nearest boundary.
        """
        bound_force = 1
        bound_vel = np.zeros(2)
        if self.pos[0] < 10:
            bound_vel[0] = bound_force
        elif self.pos[0] > self.WIDTH - 10:
            bound_vel[0] = -bound_force
        if self.pos[1] < 10:
            bound_vel[1] = bound_force
        elif self.pos[1] > self.HEIGHT - 10:
            bound_vel[1] = -bound_force

        return bound_vel
