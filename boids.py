import numpy as np
import matplotlib.pyplot as plt


class Boid:
    def __init__(self, pos: np.ndarray, vel: np.ndarray, WIDTH: int, HEIGHT: int):
        self.pos = pos
        self.vel = vel
        self.artist = None
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT

        self.bound_force = 1

        self.cohesion_weight = 1
        self.alignment_weight = 1
        self.separation_weight = 1
        self.flee_weight = 10

    def update_plot(self, ax: plt.Axes, c="b") -> plt.Artist:
        """
        Updates the position of the boid's plot and returns the artist object.
        """
        orientation = np.arctan2(self.vel[1], self.vel[0])
        self.artist, = ax.plot(self.pos[0], self.pos[1], markersize=3, marker=(
            3, 0, np.degrees(orientation) - 90), linestyle="", color=c)
        return self.artist

    def update(self, neighborhood: list, max_speed: float = 1, max_force: float = 0.5) -> None:
        """
        Updates the position and velocity of the boid based on its current acceleration.
        """
        alignment = self.align(neighborhood)
        cohesion = self.cohesion(neighborhood)
        separation = self.separation(neighborhood)
        flee = self.flee(neighborhood)
        bound = self.boundary()

        # Update the velocity
        acc = alignment * self.alignment_weight + cohesion * self.cohesion_weight + \
            separation * self.separation_weight + flee * self.flee_weight + bound
        if np.linalg.norm(acc) > max_force:
            acc = max_force * acc / np.linalg.norm(acc)
        self.vel += acc
        if np.linalg.norm(self.vel) > max_speed:
            self.vel = max_speed * self.vel / np.linalg.norm(self.vel)
        self.pos += self.vel

    def check_bounds(self, width: int, height: int) -> None:
        """F
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
        neighborhood = {d: b for d, b in zip(
            dist, boids) if d < radius and d != 0}
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
        pos_diff = np.array(
            [boid.pos - self.pos for boid in neighborhood.values()])
        dist_squared = np.array([dist**2 for dist in neighborhood.keys()])
        separation = -np.sum(pos_diff / dist_squared[:, None], axis=0)
        return separation

    def boundary(self) -> np.ndarray:
        """
        Returns a vector pointing away from the nearest boundary.
        """
        bound_vel = np.zeros(2)
        if self.pos[0] < 10:
            bound_vel[0] = self.bound_force
        elif self.pos[0] > self.WIDTH - 10:
            bound_vel[0] = -self.bound_force
        if self.pos[1] < 10:
            bound_vel[1] = self.bound_force
        elif self.pos[1] > self.HEIGHT - 10:
            bound_vel[1] = -self.bound_force

        return bound_vel

    def flee(self, neighborhood: dict) -> np.ndarray:
        """
        Returns a vector pointing away from all the predators in the neighborhood.
        """
        if len(neighborhood) == 0:
            return np.zeros(2)
        flee_vel = np.zeros(2)
        for boid in neighborhood.values():
            if isinstance(boid, Predator):
                flee_vel += (self.pos - boid.pos) / np.linalg.norm(self.pos - boid.pos)**2
        return flee_vel

class Predator(Boid):
    def __init__(self, pos: np.ndarray, vel: np.ndarray, WIDTH: int, HEIGHT: int):
        super().__init__(pos, vel, WIDTH, HEIGHT)
        self.wander_strength = 1
        self.wander_angle = 0
        self.angle_change = 0.1

    def update(self, neighborhood: list, max_speed: float = 1.5, max_force: float = 0.1) -> None:
        """
        Updates the position and velocity of the predator based on its current acceleration.
        """
        seek = self.seek(neighborhood)  # Seek the closest boid
        wander = self.wander()  # Wander around if no boids are nearby
        bound = self.boundary()

        # Change weights depending on situation
        if np.linalg.norm(seek) != 0:  # If there are boids nearby, don't wander
            wander_weight = 0
        else:
            wander_weight = 1
        if self.pos[0] < 10 or self.pos[0] > self.WIDTH - 10 or self.pos[1] < 10 or self.pos[1] > self.HEIGHT - 10:
            wander_weight = 0  # Don't wander if near the boundary

        # Update the velocity
        acc = seek + wander * wander_weight + bound
        if np.linalg.norm(acc) > max_force:
            acc = max_force * acc / np.linalg.norm(acc)
        self.vel += acc
        if np.linalg.norm(self.vel) > max_speed:
            self.vel = max_speed * self.vel / np.linalg.norm(self.vel)
        self.pos += self.vel

    def seek(self, neighborhood: dict) -> np.ndarray:
        """
        Returns the vector pointing towards the closest boid.
        """
        if len(neighborhood) == 0:
            return np.zeros(2)
        closest_boid = None
        closest_dist = np.inf
        for dist, boid in neighborhood.items():
            if dist < closest_dist and not isinstance(boid, Predator):
                closest_dist = dist
                closest_boid = boid
        return closest_boid.pos - self.pos

    def wander(self) -> np.ndarray:
        """
        Returns a vector pointing in a random direction.	
        """
        circle_center = self.vel / \
            np.linalg.norm(self.vel) * self.wander_strength
        displacement = np.array([0, -1])

        # Rotate the displacement vector by the wander angle
        displacement = np.array([displacement[0] * np.cos(self.wander_angle) - displacement[1] * np.sin(self.wander_angle),
                                 displacement[0] * np.sin(self.wander_angle) + displacement[1] * np.cos(self.wander_angle)])

        # Change the wander angle slightly
        self.wander_angle += np.random.uniform(-self.angle_change,
                                               self.angle_change)
        return circle_center + displacement

    def update_plot(self, ax: plt.Axes, c="b") -> plt.Artist:
        orientation = np.arctan2(self.vel[1], self.vel[0])
        self.artist, = ax.plot(self.pos[0], self.pos[1], markersize=3, marker=(
            3, 0, np.degrees(orientation) - 90), linestyle="", color='red')
        return self.artist
