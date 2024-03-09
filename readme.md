# Boids
This small project is an implementation of the Boids algorithm by Craig Reynolds[^1] in Python. The algorithm simulates the flocking 
behavior of birds or fish by defining three simple rules:
1. **Separation**: Boids avoid collisions with their neighbors by steering away from them.
2. **Alignment**: Boids align their velocity with the average velocity of their neighbors.
3. **Cohesion**: Boids steer towards the average position of their neighbors.
This implementation is based on the description of the Boids algorithm by Coding Train[^2].

***
## Source
[^1]: Reynolds, C. W. (1987). Flocks, herds and schools: A distributed behavioral model. ACM SIGGRAPH Computer Graphics, 21(4), 25-34.[Link to the original paper](https://www.red3d.com/cwr/papers/1987/boids.html)
[^2]: [The Coding Train](https://thecodingtrain.com/CodingChallenges/124-flocking-boids.html)