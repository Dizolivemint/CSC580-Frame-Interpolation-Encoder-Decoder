import math
import numpy as np

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

    def simulate_ball_motion(self, mass, angle, friction, time_steps=50, height=64, width=64):
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
        s = 0
        frames = []

        # Start from bottom-left of the frame
        ramp_start = (0, height - 1)

        # Compute ramp end point based on angle and frame size
        rad = math.radians(angle)
        dx = math.cos(rad)
        dy = -math.sin(rad)  # negative to go upward on image (since y=0 is top)

        # Normalize the ramp to fill the diagonal range of the frame
        length_scale = min(width / abs(dx) if dx != 0 else float('inf'),
                          height / abs(dy) if dy != 0 else float('inf'))
        
        ramp_end = (
            ramp_start[0] + dx * length_scale,
            ramp_start[1] + dy * length_scale
        )
        
        ramp_length = math.hypot(ramp_end[0] - ramp_start[0], ramp_end[1] - ramp_start[1])

        for _ in range(time_steps):
            velocity += acc * self.dt
            s += velocity * self.dt
            s = max(0, min(s, ramp_length))

            normalized = s / ramp_length

            # Compute current position along the ramp
            x = ramp_start[0] + normalized * (ramp_end[0] - ramp_start[0])
            y = ramp_start[1] + normalized * (ramp_end[1] - ramp_start[1])

            # Draw position in a blank image
            frame = np.zeros((height, width), dtype=np.float32)
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= xi < width and 0 <= yi < height:
                frame[yi, xi] = 1.0  # mark ball position
            frames.append(frame)

        return frames

    # Future function examples:
    # def simulate_camera_motion(self, ...)
    # def simulate_object_2d_motion(self, ...)
    # def simulate_projectile(self, ...)

if __name__ == "__main__":
    simulator = PhysicsSimulator(gravity=9.8, dt=0.1)
    positions = simulator.simulate_ball_motion(mass=0.15, angle=30, friction=0.05, time_steps=50)
    print(positions)