import numpy as np 
from scipy import interpolate
import random
def flat_ground_terrain(terrain, height, part=1): 
    """
    terrain (SubTerrain): the terrain
    height (float): the height of the terrain
    part (int): the added part, 1 2 3 4 
    """
    x, y = terrain.determine_part_coordinates(part)
    terrain.height_field_raw[x: x + terrain.part_width, y: y + terrain.part_length] += int (height / terrain.vertical_scale)
    return terrain


import numpy as np
from noise import pnoise2 

def natural_ground_terrain(terrain, scale=10.0, octaves=6, persistence=0.5, lacunarity=2.0, height_multiplier=5.0, part=1):
    """
    terrain (SubTerrain): the terrain
    scale (float): frequency of the noise (higher value = more detail)
    octaves (int): number of noise layers to combine
    persistence (float): amplitude multiplier for each octave
    lacunarity (float): frequency multiplier for each octave
    height_multiplier (float): scales the overall height of the terrain
    part (int): the added part, 1 2 3 4
    """
    x, y = terrain.determine_part_coordinates(part) 

    for i in range(terrain.part_width):
        for j in range(terrain.part_length):
            sample_x = (x + i) / scale
            sample_y = (y + j) / scale
            noise_value = pnoise2(sample_x, sample_y, octaves=octaves, persistence=persistence, lacunarity=lacunarity)

            height = int(noise_value * height_multiplier / terrain.vertical_scale)
            terrain.height_field_raw[x + i, y + j] += height

    return terrain


def generate_river_terrain(terrain, river_width, river_depth, river_path, bank_height, smoothness=1.0, part=1):
    """
    Generate a terrain with a river

    Parameters
        terrain (SubTerrain): the terrain object that will hold the generated terrain data.
        river_width (float): the width of the river [meters].
        river_depth (float): how much lower the river should be compared to the surrounding area [meters].
        river_path (list of tuples): a list of (x, y) coordinates defining the path of the river centerline.
        bank_height (float): the height of the banks relative to the river bottom [meters].
        smoothness (float): a factor to control the smoothness of the terrain transition from river to land.
    Returns
        SubTerrain: the modified terrain with a river.
    """
    # Convert parameters to discrete units
    x_range, y_range = terrain.determine_part_coordinates(part)
    river_width = river_width / terrain.horizontal_scale
    river_depth = river_depth / terrain.vertical_scale
    bank_height = bank_height / terrain.vertical_scale

    # Create a grid of x and y coordinates
    x_coords = np.arange(0, terrain.part_width)
    y_coords = np.arange(0, terrain.part_length)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing='ij')

    # Calculate distances from each point to the river path
    distances = []
    for x, y in river_path:
        distances.append(np.sqrt((x_grid - x) ** 2 + (y_grid - y) ** 2))
    distance_map = np.min(distances, axis=0)

    # Create a mask for the river area
    river_mask = distance_map <= river_width // 2
    # Define the height based on the distance from the river
    height_map = np.where(river_mask, -river_depth, 0)
    height_map += bank_height * (1 - np.exp(-distance_map**2 / (2 * (river_width // 2)**2 * smoothness)))

    # Apply the height map to the terrain
    terrain.height_field_raw[x_range: x_range + terrain.part_width, y_range: y_range + terrain.part_length] += height_map.astype(np.int16)
    return terrain


def random_uniform_terrain(terrain, min_height, max_height, step=1, downsampled_scale=None, part=1):
    """
    Generate a uniform noise terrain

    Parameters
        terrain (SubTerrain): the terrain
        min_height (float): the minimum height of the terrain [meters]
        max_height (float): the maximum height of the terrain [meters]
        step (float): minimum height change between two points [meters]
        downsampled_scale (float): distance between two randomly sampled points ( musty be larger or equal to terrain.horizontal_scale)

    """
    x_range, y_range = terrain.determine_part_coordinates(part)
    if downsampled_scale is None:
        downsampled_scale = terrain.horizontal_scale

    # switch parameters to discrete units
    min_height = int(min_height / terrain.vertical_scale)
    max_height = int(max_height / terrain.vertical_scale)
    step = int(step / terrain.horizontal_scale)

    heights_range = np.arange(min_height, max_height + step, step)
    height_field_downsampled = np.random.choice(heights_range, (int(terrain.part_width / downsampled_scale), int(
        terrain.part_length / downsampled_scale)))

    x = np.linspace(x_range, terrain.part_width, height_field_downsampled.shape[0])
    y = np.linspace(y_range, terrain.part_length, height_field_downsampled.shape[1])

    f = interpolate.interp2d(y, x, height_field_downsampled, kind='linear')

    x_upsampled = np.linspace(0, terrain.part_width, terrain.part_width)
    y_upsampled = np.linspace(0, terrain.part_length, terrain.part_length)
    z_upsampled = np.rint(f(y_upsampled, x_upsampled))

    terrain.height_field_raw[x_range: x_range + terrain.part_width, y_range: y_range + terrain.part_length] += z_upsampled.astype(np.int16)
    return terrain


import numpy as np

def sloped_terrain(terrain, slope=0.2, part=1):
    """
    Generate a pyramid-shaped terrain that can be ascended from all four directions.

    Parameters:
        terrain (SubTerrain): the terrain
        slope (float): positive or negative slope
        part (int): the part of the terrain to modify
    Returns:
        terrain (SubTerrain): updated terrain
    """
    x_range, y_range = terrain.determine_part_coordinates(part)
    
    # Calculate center coordinates
    x_center = (terrain.part_width - 1) / 2
    y_center = (terrain.part_length - 1) / 2
    
    # Create grid of coordinates
    x = np.arange(terrain.part_width)
    y = np.arange(terrain.part_length)
    xx, yy = np.meshgrid(x, y, indexing='ij')  # Use 'ij' indexing for matrix coordinates
    
    # Calculate distance from center using Manhattan distance
    dx = xx - x_center
    dy = yy - y_center
    distance = np.abs(dx) + np.abs(dy)  # Manhattan distance
    
    # Calculate maximum possible distance from center (corner points)
    max_distance = x_center + y_center
    
    # Calculate height parameters based on terrain scaling
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * max_distance)
    
    # Generate height field with pyramid shape
    height = (max_height * (1 - distance / max_distance)).astype(terrain.height_field_raw.dtype)
    
    # Apply height field to the terrain
    terrain.height_field_raw[x_range: x_range + terrain.part_width, 
                            y_range: y_range + terrain.part_length] += height
    
    return terrain


def pyramid_sloped_terrain(terrain, slope=0.2, platform_size=1., part=1):
    """
    Generate a sloped terrain

    Parameters:
        terrain (terrain): the terrain
        slope (int): positive or negative slope
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    x_range, y_range = terrain.determine_part_coordinates(part)
    x = np.arange(0, terrain.part_width)
    y = np.arange(0, terrain.part_length)
    center_x = int(terrain.part_width / 2)
    center_y = int(terrain.part_length / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x-xx)) / center_x
    yy = (center_y - np.abs(center_y-yy)) / center_y
    xx = xx.reshape(terrain.part_width, 1)
    yy = yy.reshape(1, terrain.part_length)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * (terrain.part_width / 2))
    terrain.height_field_raw[x_range: x_range + terrain.part_width, y_range: y_range + terrain.part_length] += (max_height * xx * yy).astype(terrain.height_field_raw.dtype)

    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.part_width // 2 - platform_size
    x2 = terrain.part_width // 2 + platform_size
    y1 = terrain.part_length // 2 - platform_size
    y2 = terrain.part_length // 2 + platform_size

    min_h = min(terrain.height_field_raw[x1 + x_range, y1 + y_range], 0)
    max_h = max(terrain.height_field_raw[x1 + x_range, y1 + y_range], 0)
    terrain.height_field_raw[x_range: x_range + terrain.part_width, y_range: y_range + terrain.part_length] = np.clip(terrain.height_field_raw[x_range: x_range + terrain.part_width, y_range: y_range + terrain.part_length], min_h, max_h)
    return terrain


def discrete_obstacles_terrain(terrain, max_height, min_size, max_size, num_rects, platform_size=1., part=1):
    """
    Generate a terrain with gaps

    Parameters:
        terrain (terrain): the terrain
        max_height (float): maximum height of the obstacles (range=[-max, -max/2, max/2, max]) [meters]
        min_size (float): minimum size of a rectangle obstacle [meters]
        max_size (float): maximum size of a rectangle obstacle [meters]
        num_rects (int): number of randomly generated obstacles
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    x_range, y_range = terrain.determine_part_coordinates(part)
    max_height = int(max_height / terrain.vertical_scale)
    min_size = int(min_size / terrain.horizontal_scale)
    max_size = int(max_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    (i, j) = terrain.part_width, terrain.part_length
    height_range = [-max_height, -max_height // 2, max_height // 2, max_height]
    width_range = range(min_size, max_size, 4)
    length_range = range(min_size, max_size, 4)

    for _ in range(num_rects):
        width = np.random.choice(width_range)
        length = np.random.choice(length_range)
        start_i = np.random.choice(range(0, i-width, 4))
        # print(j, length)
        start_j = np.random.choice(range(0, j-length, 4))
        terrain.height_field_raw[x_range + start_i: x_range + start_i+width, y_range + start_j:y_range + start_j+length] = np.random.choice(height_range)

    return terrain


def wave_terrain(terrain, num_waves=1, amplitude=1., part=1):
    """
    Generate a wavy terrain

    Parameters:
        terrain (terrain): the terrain
        num_waves (int): number of sine waves across the terrain length
    Returns:
        terrain (SubTerrain): update terrain
    """
    x_range, y_range = terrain.determine_part_coordinates(part)
    amplitude = int(0.5*amplitude / terrain.vertical_scale)
    if num_waves > 0:
        div = terrain.part_length / (num_waves * np.pi * 2)
        x = np.arange(0, terrain.part_width)
        y = np.arange(0, terrain.part_length)
        xx, yy = np.meshgrid(x, y, sparse=True)
        xx = xx.reshape(terrain.part_width, 1)
        yy = yy.reshape(1, terrain.part_length)
        terrain.height_field_raw[x_range: x_range + terrain.part_width, y_range: y_range + terrain.part_length] += (amplitude*np.cos(yy / div) + amplitude*np.sin(xx / div)).astype(
            terrain.height_field_raw.dtype)
    return terrain


def stairs_terrain(terrain, step_width, step_height, part=1):
    """
    Generate a stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float):  the height of the step [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    x_range, y_range = terrain.determine_part_coordinates(part)
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)

    num_steps = terrain.part_width // step_width
    height = step_height
    for i in range(num_steps):
        terrain.height_field_raw[x_range + i * step_width: (i + 1) * step_width, y_range: y_range + terrain.part_length] += height
        height += step_height
    return terrain


def pyramid_stairs_terrain(terrain, step_width, step_height, platform_size=1., part=1):
    """
    Generate stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float): the step_height [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    x_range, y_range = terrain.determine_part_coordinates(part)
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    height = 0
    start_x = x_range
    stop_x = terrain.part_width
    start_y = y_range
    stop_y = terrain.part_length
    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        height += step_height
        terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height
    return terrain


def gap_terrain(terrain, gap_size, platform_size=1., part=1, depth=1000):
    x_range, y_range = terrain.determine_part_coordinates(part)
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.part_length // 2
    center_y = terrain.part_width // 2
    x1 = (terrain.part_length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.part_width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[center_x-x2 + x_range : center_x + x2 + x_range, center_y-y2 + y_range : center_y + y2 + y_range] = -depth
    return terrain

def pillars_terrain(terrain, num_pillars=3, max_pillar_size=2.0, pillar_gap=2.0, step_height=0.2, part=1):
    """
    Generate a terrain with pillars of different shapes.

    Parameters:
        terrain (Terrain): the terrain
        num_pillars (int): number of pillars
        max_pillar_size (float): maximum size of the pillar (meters)
        pillar_gap (float): gap between pillars (meters)
        step_height (float): height of the step (meters)
        part (int): part of the terrain to modify

    Returns:
        terrain (SubTerrain): updated terrain
    """
    x_range, y_range = terrain.determine_part_coordinates(part)
    max_pillar_size = int(max_pillar_size / terrain.horizontal_scale)
    pillar_height = int(1.5 / terrain.vertical_scale)
    step_height = int(step_height / terrain.vertical_scale)
    pillar_gap = int(pillar_gap / terrain.horizontal_scale)

    # Set step-like borders around the terrain part
    block_length = int(terrain.part_length / 3)
    block_width = int(terrain.part_width / 3)
    
    # Set borders with step_height
    terrain.height_field_raw[x_range, y_range:y_range + terrain.part_length] = step_height
    terrain.height_field_raw[x_range + block_length:x_range + block_length + 2, y_range:y_range + terrain.part_length] = step_height
    terrain.height_field_raw[x_range + 2*block_length:x_range + 2*block_length + 2, y_range:y_range + terrain.part_length] = step_height
    terrain.height_field_raw[x_range + terrain.part_width - 1, y_range:y_range + terrain.part_length] = step_height
    
    terrain.height_field_raw[x_range:x_range + terrain.part_width, y_range] = step_height
    terrain.height_field_raw[x_range:x_range + terrain.part_width, y_range + block_width:y_range + block_width + 2] = step_height
    terrain.height_field_raw[x_range:x_range + terrain.part_width, y_range + 2*block_width:y_range + 2*block_width + 2] = step_height
    terrain.height_field_raw[x_range:x_range + terrain.part_width, y_range + terrain.part_length - 1] = step_height

    center_x = terrain.part_width // 2
    center_y = terrain.part_length // 2
    pillar_count = 0
    num_attempts = 0

    while pillar_count < num_pillars and num_attempts < 100:
        # Random position and size
        x = random.randint(0, terrain.part_width - max_pillar_size)
        y = random.randint(0, terrain.part_length - max_pillar_size)
        w = int(random.uniform(0.5, 1) * max_pillar_size)
        h = int(random.uniform(0.5, 1) * max_pillar_size)
        
        # Calculate extended area considering pillar_gap
        x_start = max(0, x - pillar_gap)
        y_start = max(0, y - pillar_gap)
        x_end = min(terrain.part_width, x + w + pillar_gap)
        y_end = min(terrain.part_length, y + h + pillar_gap)
        
        # Check if the extended area is free
        area = terrain.height_field_raw[x_start + x_range:x_end + x_range, y_start + y_range:y_end + y_range]
        if not np.all(area < pillar_height):
            num_attempts += 1
            continue
        
        # Check if not in central area
        if not ((x + w < center_x - 1) or (x > center_x + 1) or (y + h < center_y - 1) or (y > center_y + 1)):
            num_attempts += 1
            continue
        
        # Place the pillar
        terrain.height_field_raw[x + x_range:x + w + x_range, y + y_range:y + h + y_range] = pillar_height
        pillar_count += 1
        num_attempts = 0  # Reset attempts after successful placement

    return terrain


from scipy.ndimage import gaussian_filter1d

class SubTerrain:
    def __init__(self, terrain_name="terrain", width=256, length=256, vertical_scale=1.0, horizontal_scale=1.0):
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        self.w1 = 0
        self.w2 = self.width // 2
        self.l1 = 0 
        self.l2 = self.length // 2
        self.part_width = self.width // 2
        self.part_length = self.length // 2
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)

    def determine_part_coordinates(self, part):
        """
        part: {1, 2, 3, 4}
        """
        x = self.w1
        y = self.l1
        if part == 2:
            x = self.w2
        elif part == 3: 
            y = self.l2
        elif part == 4:
            x = self.w2
            y = self.l2
        return (x, y)
    
    def _apply_horizontal_filter(self, x_start, x_end, y_start, y_end, sigma):
        transition = self.height_field_raw[x_start:x_end, y_start:y_end]
        if transition.size > 0:
            smoothed = gaussian_filter1d(transition, sigma=sigma, axis=0, mode='nearest')
            self.height_field_raw[x_start:x_end, y_start:y_end] = smoothed

    def _apply_vertical_filter(self, x_start, x_end, y_start, y_end, sigma):
        transition = self.height_field_raw[x_start:x_end, y_start:y_end]
        if transition.size > 0:
            smoothed = gaussian_filter1d(transition, sigma=sigma, axis=1, mode='nearest')
            self.height_field_raw[x_start:x_end, y_start:y_end] = smoothed

    def smooth_transitions(self, n=5, sigma=2.0):
        h_trans1_x_start = self.w1 + self.part_width - n
        h_trans1_x_end = self.w2 + n
        self._apply_horizontal_filter(h_trans1_x_start, h_trans1_x_end, 
                                    self.l1, self.l1 + self.part_length, 
                                    sigma)
        
        h_trans2_x_start = self.w1 + self.part_width - n
        h_trans2_x_end = self.w2 + n
        self._apply_horizontal_filter(h_trans2_x_start, h_trans2_x_end,
                                    self.l2, self.l2 + self.part_length,
                                    sigma)

        v_trans1_y_start = self.l1 + self.part_length - n
        v_trans1_y_end = self.l2 + n
        self._apply_vertical_filter(self.w1, self.w1 + self.part_width,
                                    v_trans1_y_start, v_trans1_y_end,
                                    sigma)
        
        v_trans2_y_start = self.l1 + self.part_length - n
        v_trans2_y_end = self.l2 + n
        self._apply_vertical_filter(self.w2, self.w2 + self.part_width,
                                    v_trans2_y_start, v_trans2_y_end,
                                    sigma)
