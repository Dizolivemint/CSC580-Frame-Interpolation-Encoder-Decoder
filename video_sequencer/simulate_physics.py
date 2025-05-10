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
        coords = []

        # Start from top-left of the frame
        ramp_start = (0, 0)

        # Compute ramp end point based on angle and frame size
        rad = math.radians(angle)
        dx = math.cos(rad)     # → right
        dy = math.sin(rad)  

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
            coord = np.zeros((height, width), dtype=np.float32)
            
            # Convert to integer coordinates and ensure they're within bounds
            xi = int(round(min(max(x, 0), width - 1)))
            yi = int(round(min(max(y, 0), height - 1)))
            coord[yi, xi] = 1.0
            coords.append(coord)

        return coords

    def simulate_projectile_motion(self, initial_velocity, angle, gravity, time_steps):
        """
        Simulate projectile motion of a particle launched with an initial velocity and angle.

        Args:
            initial_velocity (float): initial speed in m/s
            angle (float): launch angle in degrees
            gravity (float): gravitational constant (usually 9.8)
            time_steps (int): number of simulation steps

        Returns:
            List of coordinates where each coordinate is a 2D array with a single point marked as 1.0
        """
        angle_rad = np.deg2rad(angle)
        v_x = initial_velocity * np.cos(angle_rad)
        v_y = initial_velocity * np.sin(angle_rad)

        coords = []
        x, y = 0.0, 0.0
        for t in range(time_steps):
            # Debug
            # print(f"t={t}: BEFORE x={x:.2f}, y={y:.2f}")
            
            # Normalize to fit in 64x64 frame (arbitrary scaling)
            x_norm = min(max(x / 100.0, 0.0), 1.0)  # scale x max to ~100 meters
            y_norm = min(max(y / 100.0, 0.0), 1.0)  # scale y max to ~100 meters
            
            # stop when object hits the an edge of the frame
            if y_norm < 0 or x_norm > 1.0 or x_norm < 0.0 or y_norm > 1.0:
                break
              
            # Create a coordinate frame with the point marked
            coord = np.zeros((64, 64), dtype=np.float32)
            xi = int(round(x_norm * 63))
            yi = int(round((1.0 - y_norm) * 63))  # invert y for image coordinates
            if 0 <= xi < 64 and 0 <= yi < 64:
                coord[yi, xi] = 1.0
            coords.append(coord)
            
            # Debug
            # print(f"t={t}: x={x:.2f}, y={y:.2f}, v_x={v_x:.2f}, v_y={v_y:.2f}")
            
            # Update position
            x += v_x * self.dt
            y += v_y * self.dt
            v_y -= gravity * self.dt  # gravity reduces vertical velocity

        # Pad to maintain consistent sequence length
        while len(coords) < time_steps:
            coords.append(np.zeros((64, 64), dtype=np.float32))

        return coords

if __name__ == "__main__":
    simulator = PhysicsSimulator(gravity=9.8, dt=0.1)
    positions = simulator.simulate_ball_motion(mass=0.15, angle=30, friction=0.05, time_steps=50)
    print(positions)