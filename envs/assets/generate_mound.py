import numpy as np
import struct

NUM_COLUMNS = 100
NUM_ROWS = 100
FILE = 'nums.txt'

def compute_rise(x, y): 
    return max(0, -(10/(12 * 9**2)) * (x**2 + y**2) + (10 / 12))

def generate_hills():
    # setup coordinate grid
    matrix = np.zeros((NUM_ROWS, NUM_COLUMNS))
    
    # Discretize the grid based on the dimensions of the baseball mound, a 
    # 18 foot diameter circle with the center being 10 inches off the ground.
    delta_x = 18 / (NUM_COLUMNS - 1)
    delta_y = 18 / (NUM_ROWS - 1)

    # Compute the rise for each point.
    for i in range(NUM_ROWS): 
        y = -9 + (i * delta_y)
        for j in range(NUM_COLUMNS): 
            x = -9 + (j * delta_x)
            matrix[i][j] = compute_rise(x, y)

    return matrix

def main():
    if NUM_COLUMNS % 2 or NUM_ROWS % 2: 
        raise ValueError('Number of rows and columns must be even.')

    matrix = generate_hills()

    with open(FILE, 'wb') as writer: 
        writer.write(struct.pack('i', NUM_ROWS))
        writer.write(struct.pack('i', NUM_COLUMNS))
        for row in matrix: 
            for val in row: 
                writer.write(struct.pack('f', val))

if __name__ == "__main__":
    main()
