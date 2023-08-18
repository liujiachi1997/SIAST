from trainer import run_training_loop, self_training_loop
import sys


config_file = sys.argv[1]
self_training_loop(config_file)
