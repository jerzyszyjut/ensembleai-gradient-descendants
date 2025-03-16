import base64
import io
import json
import numpy as np
import onnxruntime as ort
import pickle
import requests
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from transform_configs import get_random_resized_crop_config, get_jitter_color_config
import os
from torch.utils.data import Dataset
from typing import Tuple, List
import time
from scipy.optimize import curve_fit


def exp_cost_function(epsilon, delta, alpha, beta):
    return delta * (np.exp(np.log(alpha/delta) * epsilon * beta) - 1)

