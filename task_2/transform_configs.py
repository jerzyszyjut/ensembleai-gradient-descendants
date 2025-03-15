import random


def get_random_resized_crop_config():
    return {
        'size': (32, 32),
        'scale': (0.08, 1.0),
        'ratio': (0.75, 1.333), 
    }

def get_jitter_color_config():
    return {
        'brightness': random.uniform(0.5, 2.0),
        'contrast': random.uniform(0.5, 2.0),
        'saturation': random.uniform(0.5, 2.0),
        'hue': random.uniform(0, 0.5)
    }