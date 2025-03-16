import numpy as np 
from scipy import interpolate
import random
from noise import pnoise2 
from scipy.ndimage import gaussian_filter1d

def flat_ground_terrain(terrain, height): 
    """
    terrain (SubTerrain): the terrain
    height (float): the height of the terrain
    """
    terrain.height_field_raw[:, :] += int(height / terrain.vertical_scale)
    return terrain

def natural_ground_terrain(terrain, scale=10.0, octaves=6, persistence=0.5, lacunarity=2.0, height_multiplier=5.0):
    """
    terrain (SubTerrain): the terrain
    scale (float): frequency of the noise (higher value = more detail)
    octaves (int): number of noise layers to combine
    persistence (float): amplitude multiplier for each octave
    lacunarity (float): frequency multiplier for each octave
    height_multiplier (float): scales the overall height of the terrain
    """
    for i in range(terrain.width):
        for j in range(terrain.length):
            sample_x = i / scale
            sample_y = j / scale
            noise_value = pnoise2(sample_x, sample_y, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
            height = int(noise_value * height_multiplier / terrain.vertical_scale)
            terrain.height_field_raw[i, j] += height
    return terrain

def generate_river_terrain(terrain, river_width, river_depth, river_path, bank_height, smoothness=1.0):
    """
    Generate a terrain with a river
    Parameters:
        terrain (SubTerrain): the terrain object that will hold the generated terrain data.
        river_width (float): the width of the river [meters].
        river_depth (float): how much lower the river should be compared to the surrounding area [meters].
        river_path (list of tuples): a list of (x, y) coordinates defining the path of the river centerline.
        bank_height (float): the height of the banks relative to the river bottom [meters].
        smoothness (float): a factor to control the smoothness of the terrain transition from river to land.
    """
    river_width = int(river_width / terrain.horizontal_scale)
    river_depth = int(river_depth / terrain.vertical_scale)
    bank_height = int(bank_height / terrain.vertical_scale)

    x_coords = np.arange(0, terrain.width)
    y_coords = np.arange(0, terrain.length)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing='ij')

    distances = []
    for x, y in river_path:
        distances.append(np.sqrt((x_grid - x) ** 2 + (y_grid - y) ** 2))
    distance_map = np.min(distances, axis=0)

    river_mask = distance_map <= river_width // 2
    height_map = np.where(river_mask, -river_depth, 0)
    height_map = height_map + bank_height * (1 - np.exp(-distance_map**2 / (2 * (river_width // 2)**2 * smoothness)))

    terrain.height_field_raw[:, :] += height_map.astype(np.int16)
    return terrain

def random_uniform_terrain(terrain, min_height, max_height, step=1, downsampled_scale=None):
    """
    Generate a uniform noise terrain
    Parameters:
        terrain (SubTerrain): the terrain
        min_height (float): the minimum height of the terrain [meters]
        max_height (float): the maximum height of the terrain [meters]
        step (float): minimum height change between two points [meters]
        downsampled_scale (float): distance between two randomly sampled points ( must be larger or equal to terrain.horizontal_scale)
    """
    if downsampled_scale is None:
        downsampled_scale = terrain.horizontal_scale

    min_height = int(min_height / terrain.vertical_scale)
    max_height = int(max_height / terrain.vertical_scale)
    step = int(step / terrain.horizontal_scale)

    heights_range = np.arange(min_height, max_height + step, step)
    height_field_downsampled = np.random.choice(
        heights_range, 
        (int(terrain.width / downsampled_scale), int(terrain.length / downsampled_scale))
    )

    x = np.linspace(0, terrain.width, height_field_downsampled.shape[0])
    y = np.linspace(0, terrain.length, height_field_downsampled.shape[1])

    f = interpolate.interp2d(y, x, height_field_downsampled, kind='linear')
    x_upsampled = np.linspace(0, terrain.width, terrain.width)
    y_upsampled = np.linspace(0, terrain.length, terrain.length)
    z_upsampled = np.rint(f(y_upsampled, x_upsampled))

    terrain.height_field_raw[:, :] += z_upsampled.astype(np.int16)
    return terrain

def sloped_terrain(terrain, slope=0.2):
    """
    Generate a pyramid-shaped terrain that can be ascended from all four directions.
    Parameters:
        terrain (SubTerrain): the terrain
        slope (float): positive or negative slope
    """
    x_center = (terrain.width - 1) / 2
    y_center = (terrain.length - 1) / 2
    x = np.arange(terrain.width)
    y = np.arange(terrain.length)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    dx = xx - x_center
    dy = yy - y_center
    distance = np.abs(dx) + np.abs(dy)
    max_distance = x_center + y_center
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * max_distance)
    height = (max_height * (1 - distance / max_distance)).astype(terrain.height_field_raw.dtype)
    terrain.height_field_raw[:, :] += height
    return terrain

def pyramid_sloped_terrain(terrain, slope=0.2, platform_size=1.):
    """
    Generate a sloped terrain
    Parameters:
        terrain (terrain): the terrain
        slope (int): positive or negative slope
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    """
    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    center_x = int(terrain.width / 2)
    center_y = int(terrain.length / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y
    xx = xx.reshape(terrain.width, 1)
    yy = yy.reshape(1, terrain.length)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * (terrain.width / 2))
    terrain.height_field_raw[:, :] += (max_height * xx * yy).astype(terrain.height_field_raw.dtype)

    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.width // 2 - platform_size
    x2 = terrain.width // 2 + platform_size
    y1 = terrain.length // 2 - platform_size
    y2 = terrain.length // 2 + platform_size

    min_h = min(terrain.height_field_raw[x1, y1], 0)
    max_h = max(terrain.height_field_raw[x1, y1], 0)
    terrain.height_field_raw[:, :] = np.clip(terrain.height_field_raw[:, :], min_h, max_h)
    return terrain

def discrete_obstacles_terrain(terrain, max_height, min_size, max_size, num_rects, platform_size=1.):
    """
    Generate a terrain with gaps
    Parameters:
        terrain (terrain): the terrain
        max_height (float): maximum height of the obstacles (range=[-max, -max/2, max/2, max]) [meters]
        min_size (float): minimum size of a rectangle obstacle [meters]
        max_size (float): maximum size of a rectangle obstacle [meters]
        num_rects (int): number of randomly generated obstacles
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    """
    max_height = int(max_height / terrain.vertical_scale)
    min_size = int(min_size / terrain.horizontal_scale)
    max_size = int(max_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    height_range = [-max_height, -max_height // 2, max_height // 2, max_height]
    width_range = range(min_size, max_size, 4)
    length_range = range(min_size, max_size, 4)

    for _ in range(num_rects):
        width = np.random.choice(width_range)
        length = np.random.choice(length_range)
        start_i = np.random.choice(range(0, terrain.width - width, 4))
        start_j = np.random.choice(range(0, terrain.length - length, 4))
        terrain.height_field_raw[start_i:start_i+width, start_j:start_j+length] = np.random.choice(height_range)
    return terrain

def wave_terrain(terrain, num_waves=1, amplitude=1.):
    """
    Generate a wavy terrain
    Parameters:
        terrain (terrain): the terrain
        num_waves (int): number of sine waves across the terrain length
    """
    amplitude = int(0.5 * amplitude / terrain.vertical_scale)
    if num_waves > 0:
        div = terrain.length / (num_waves * np.pi * 2)
        x = np.arange(0, terrain.width)
        y = np.arange(0, terrain.length)
        xx, yy = np.meshgrid(x, y, sparse=True)
        xx = xx.reshape(terrain.width, 1)
        yy = yy.reshape(1, terrain.length)
        terrain.height_field_raw[:, :] += (amplitude * np.cos(yy / div) + amplitude * np.sin(xx / div)).astype(
            terrain.height_field_raw.dtype)
    return terrain

def stairs_terrain(terrain, step_width, step_height):
    """
    Generate stairs
    Parameters:
        terrain (terrain): the terrain
        step_width (float): the width of the step [meters]
        step_height (float): the height of the step [meters]
    """
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)

    num_steps = terrain.width // step_width
    height = step_height
    for i in range(num_steps):
        terrain.height_field_raw[i * step_width: (i + 1) * step_width, :] += height
        height += step_height
    return terrain

def pyramid_stairs_terrain(terrain, step_width, step_height, platform_size=1.):
    """
    Generate stairs
    Parameters:
        terrain (terrain): the terrain
        step_width (float): the width of the step [meters]
        step_height (float): the step_height [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    """
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    height = 0
    start_x, stop_x = 0, terrain.width
    start_y, stop_y = 0, terrain.length
    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        height += step_height
        terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = height
    return terrain

def gap_terrain(terrain, gap_size, platform_size=1., depth=1000):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[center_x - x2: center_x + x2, center_y - y2: center_y + y2] = -depth
    return terrain

def pillars_terrain(terrain, num_pillars=3, max_pillar_size=2.0, pillar_gap=2.0, step_height=0.2):
    """
    Generate a terrain with pillars of different shapes.
    Parameters:
        terrain (Terrain): the terrain
        num_pillars (int): number of pillars
        max_pillar_size (float): maximum size of the pillar (meters)
        pillar_gap (float): gap between pillars (meters)
        step_height (float): height of the step (meters)
    """
    max_pillar_size = int(max_pillar_size / terrain.horizontal_scale)
    pillar_height = int(1.5 / terrain.vertical_scale)
    step_height = int(step_height / terrain.vertical_scale)
    pillar_gap = int(pillar_gap / terrain.horizontal_scale)

    terrain.height_field_raw[0, :] = step_height
    terrain.height_field_raw[-1, :] = step_height
    terrain.height_field_raw[:, 0] = step_height
    terrain.height_field_raw[:, -1] = step_height

    center_x = terrain.width // 2
    center_y = terrain.length // 2
    pillar_count = 0
    num_attempts = 0

    while pillar_count < num_pillars and num_attempts < 100:
        x = random.randint(0, terrain.width - max_pillar_size)
        y = random.randint(0, terrain.length - max_pillar_size)
        w = int(random.uniform(0.5, 1) * max_pillar_size)
        h = int(random.uniform(0.5, 1) * max_pillar_size)
        
        x_start = max(0, x - pillar_gap)
        y_start = max(0, y - pillar_gap)
        x_end = min(terrain.width, x + w + pillar_gap)
        y_end = min(terrain.length, y + h + pillar_gap)
        
        area = terrain.height_field_raw[x_start:x_end, y_start:y_end]
        if not np.all(area < pillar_height):
            num_attempts += 1
            continue
        
        if not ((x + w < center_x - 1) or (x > center_x + 1) or (y + h < center_y - 1) or (y > center_y + 1)):
            num_attempts += 1
            continue
        
        terrain.height_field_raw[x:x + w, y:y + h] = pillar_height
        pillar_count += 1
        num_attempts = 0

    return terrain

class SubTerrain:
    def __init__(self, terrain_name="terrain", width=256, length=256, vertical_scale=1.0, horizontal_scale=1.0):
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)

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
        self._apply_horizontal_filter(0, self.width, 0, self.length, sigma)
        self._apply_vertical_filter(0, self.width, 0, self.length, sigma)