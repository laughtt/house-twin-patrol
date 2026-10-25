"""Where is the duck: one module per job, run by agent/localize.py's ParticleFilter each tick.

housemap   the map at sensor height, distance map, ray cast, spin-match library
tof        the 8x8 ToF frame → 8 zone ranges; how well each guess explains them
spin       one full turn on the spot → match the stacked scan against the whole map → re-seed the guesses
fit        verify before trust: do the measured zones fit the map from the estimate?
landmarks  recognised objects, room colour and ArUco markers
"""
import numpy as np

CELL = 0.02
TOF_AZ = -np.radians((np.arange(8) + 0.5) / 8 * 45 - 22.5)  # zone columns, col 0 = the sensor's left (+)
TOF_MAX = 4.0  # the simulated VL53L5CX; a real one may see less (ParticleFilter tof_max)


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi
