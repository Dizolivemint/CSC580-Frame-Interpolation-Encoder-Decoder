def create_prompt(position, frame_number):
    return f"Frame {frame_number}: A ball rolling down a 30-degree ramp, currently at {position:.2f} meters."

def generate_prompts(positions):
    prompts = []
    for i, pos in enumerate(positions):
        prompts.append(create_prompt(pos, i))
    return prompts