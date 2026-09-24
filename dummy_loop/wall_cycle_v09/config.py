"""Configuration for the isolated v0.9r12 environment."""
from .clean_env import config as legacy_config

def config():
 c=legacy_config();c.width_m=c.height_m=.5;c.score_width_m=c.score_height_m=.2
 c.base_steps=60000;c.max_steps=100000;c.max_cycles=2000
 return c
