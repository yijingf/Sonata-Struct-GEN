import os
import sys

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(FILE_DIR)

sys.path.append(FILE_DIR)
sys.path.append(os.path.dirname(PARENT_DIR))
