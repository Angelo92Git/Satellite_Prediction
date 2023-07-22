import numpy as np

# Scoring functions

def smape(y1, y2):
    numerator = np.abs(y1 - y2)
    denominator = (np.abs(y1) + np.abs(y2))
    denominator[denominator == 0] = 1
    smape = np.mean(numerator / denominator)
    return smape

def vector_smape(y1, y2):
    pos1 = y1[:, :3]
    pos2 = y2[:, :3]
    pos1_norm = np.linalg.norm(pos1, axis=1)
    pos2_norm = np.linalg.norm(pos2, axis=1)
    pos_difference_norm = np.linalg.norm(pos1 - pos2, axis=1)
    pos_numerator = pos_difference_norm
    pos_denominator = (pos1_norm + pos2_norm)
    pos_denominator[pos_denominator == 0] = 1
    pos_score = pos_numerator / pos_denominator
    vel1 = y1[:, 3:]
    vel2 = y2[:, 3:]
    vel1_norm = np.linalg.norm(vel1, axis=1)
    vel2_norm = np.linalg.norm(vel2, axis=1)
    vel_difference_norm = np.linalg.norm(vel1 - vel2, axis=1)
    vel_numerator = vel_difference_norm
    vel_denominator = (vel1_norm + vel2_norm)
    vel_denominator[vel_denominator == 0] = 1
    vel_score = vel_numerator / vel_denominator
    score = (pos_score + vel_score)/2 # The position and velocity scores are equally weighted, and the each of these are normalized.
    score = np.mean(score)
    return score