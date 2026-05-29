
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    # TODO: Add the odometry factor between X(4) and X(5) to the graph (BetweenFactorPose2)
    # TODO: Based on the odometry, find the initial estimate for the pose of X(5) and add it to the graph
    
    # Get X(3) from the initial estimate
    pose_3 = initial_estimate.atPose2(X(3))
    
    # The robot rotates 45 degrees and moves ~2 meters, then rotates another 45 degrees
    # This results in X(4) being at position (4 + sqrt(2), sqrt(2)) with theta = pi/2
    pose_4_expected = gtsam.Pose2(4.0 + math.sqrt(2), math.sqrt(2), math.pi / 2)
    
    # Compute the relative pose (odometry) using the between() method
    # This gives us the transform from pose_3's frame to pose_4
    odometry = pose_3.between(pose_4_expected)
    
    # Add the odometry factor between X(3) and X(4)
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry, ODOMETRY_NOISE))
    
    # Insert the expected pose for X(4) into the initial estimate
    initial_estimate.insert(X(4), pose_4_expected)
    
    return graph, initial_estimate