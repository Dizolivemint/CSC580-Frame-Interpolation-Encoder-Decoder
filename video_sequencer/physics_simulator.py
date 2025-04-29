import math

def simulate_ball_motion(mass, angle, friction, time_steps=10, dt=0.1):
    g = 9.8  # Gravity
    acc = g * math.sin(math.radians(angle)) - friction * g * math.cos(math.radians(angle))
    velocity = 0
    position = 0
    positions = []

    for _ in range(time_steps):
        velocity += acc * dt
        position += velocity * dt
        positions.append(position)

    return positions

# Example usage
mass = 0.15  # kg
angle = 30  # degrees
friction = 0.05

positions = simulate_ball_motion(mass, angle, friction)
print(positions)
