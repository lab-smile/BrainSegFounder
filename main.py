import argparse

import numpy as np

parser = argparse.ArgumentParser(description="GatorBrain Pretraining Script")
# General
parser.add_argument("--")

def main():
    args = parser.parse_args()
    main_worker(args=args)

def main_worker(args):
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress= True)
