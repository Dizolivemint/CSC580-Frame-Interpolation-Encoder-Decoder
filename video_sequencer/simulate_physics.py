import math

class PhysicsSimulator:
    def __init__(self, gravity=9.8, dt=0.1):
        """
        Initialize the physics simulator environment.

        Args:
            gravity (float): Gravitational acceleration (m/s^2)
            dt (float): Time step size in seconds
        """
        self.gravity = gravity
        self.dt = dt

    def simulate_ball_motion(self, mass, angle, friction, time_steps=10):
        """
        Simulate ball rolling down a ramp.

        Args:
            mass (float): Mass of ball (currently not affecting result)
            angle (float): Angle of ramp in degrees
            friction (float): Coefficient of friction
            time_steps (int): Number of timesteps

        Returns:
            List of positions over time.
        """
        acc = self.gravity * math.sin(math.radians(angle)) - friction * self.gravity * math.cos(math.radians(angle))
        velocity = 0
        position = 0
        positions = []

        for _ in range(time_steps):
            velocity += acc * self.dt
            position += velocity * self.dt
            positions.append(position)

        return positions

    # Future function examples:
    # def simulate_camera_motion(self, ...)
    # def simulate_object_2d_motion(self, ...)
    # def simulate_projectile(self, ...)

if __name__ == "__main__":
    simulator = PhysicsSimulator(gravity=9.8, dt=0.1)
    positions = simulator.simulate_ball_motion(mass=0.15, angle=30, friction=0.05, time_steps=5)
    print(positions)