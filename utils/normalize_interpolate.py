def normalize_interpolate(ramp_start, ramp_end, normalized):
    """
    Get the interpolated (x, y) along a ramp given a normalized value [0.0, 1.0].

    Args:
        ramp_start (tuple): (x0, y0)
        ramp_end (tuple): (x1, y1)
        normalized (float): 0.0 (start) to 1.0 (end) position along the ramp

    Returns:
        (x, y): interpolated pixel coordinates
    """
    dx = ramp_end[0] - ramp_start[0]
    dy = ramp_end[1] - ramp_start[1]
    x = ramp_start[0] + normalized * dx
    y = ramp_start[1] + normalized * dy
    return (x, y)