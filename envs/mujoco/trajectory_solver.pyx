from libcpp.vector cimport vector
from libc.math cimport M_PI

from cpython cimport array

# Constants. 
cdef float rho = 1.23 # kg/m^3
cdef float radius = .0365 # m
cdef float drag_coef = 0.4
cdef float area = M_PI * (radius ** 2) # m^2
cdef float mass = 0.145 # kg
cdef float g_const = 9.8 # m/s^2

cdef float magnus_constant = (4/3) * M_PI * (radius ** 3)
cdef float drag_constant = (1/2) * drag_coef * rho * area

cdef float distance_to_home = 18.29 # m
cdef float spin_rate = 228.8 # rad/s


# delta
cdef float delta_t = 0.01 # s

cdef float l2_norm(float[3]& vector): 
    return (vector[0]**2 + vector[1]**2 + vector[2]**2) ** (1/2)

cdef void magnus(float[3]& ang_velo, float[3]& lin_velo, float[3]& acc): 
    acc[0] += ((ang_velo[2] * lin_velo[3]) - (ang_velo[3] * lin_velo[2])) * magnus_constant
    acc[1] += ((ang_velo[3] * lin_velo[1]) - (ang_velo[1] * lin_velo[3])) * magnus_constant
    acc[2] += ((ang_velo[1] * lin_velo[2]) - (ang_velo[2] * lin_velo[1])) * magnus_constant

"""
cdef float[3] unit(float[3]& vector, float norm): 
    cdef float[3] unit_vec = [vector[0] / norm, vector[1] / norm, vector[2] / norm]
    return unit_vec
"""

cdef void drag(float[3]& lin_velo, float[3]& acc): 
    cdef float norm = l2_norm(lin_velo)
    constant = -drag_constant * norm
    acc[0] += constant * lin_velo[0]
    acc[1] += constant * lin_velo[1]
    acc[2] += constant * lin_velo[2]

cdef void gravity(float[3]& acc): 
    acc[2] += -mass * g_const

cdef void step(float[3]& ang_velo, float[3]& lin_velo, float[3]& acc, float[3]& lin_pos): 
    magnus(ang_velo, lin_velo, acc)
    drag(lin_velo, acc)
    gravity(acc)

    for i in range(3): 
        acc[i] /= mass
        lin_pos[i] += (lin_velo[i] * delta_t) + (0.5 * acc[i] * delta_t**2)
        lin_velo[i] += acc[i] * delta_t

def compute_trajectory(list starting_velo, list starting_pos): 
    """
    This function (and associated functions) numerically integrates the kinematic equations, accounting for drag, gravitational, and magnus forces on 
    a baseball. 
    """
    cdef float[3] acc = [0, 0, 0]
    cdef float[3] ang_velo = [0, spin_rate, 0]
    cdef float[3] delta_pos = [0, 0, 0]
    cdef float time_elapsed = 0

    cdef array.array pos = array.array('f', starting_pos)
    cdef float[3] lin_pos = pos.data.as_floats

    cdef array.array velo = array.array('f', starting_velo)
    cdef float[3] lin_velo = velo.data.as_floats

    cdef float[3] initial_position = lin_pos

    while l2_norm(delta_pos) < distance_to_home: 
        step(ang_velo, lin_velo, acc, lin_pos)

        # Reset the acceleration vector and compute delta. 
        for i in range(3): 
            acc[i] = 0
            delta_pos[i] = pos[i] - initial_position[i]

        time_elapsed += delta_t

        # Break if any position falls below 0. 
        for i in range(3): 
            if lin_pos[i] < 0:
                break

    final_pos, final_velo = [], []
    for i in range(3): 
        final_pos.append(pos[i])
        final_velo.append(velo[i])
    return (time_elapsed, final_pos, final_velo)
