#!/usr/bin/env python

import numpy as np
import copy
import sys

sys.path.append("../../")

from spotmicro.GymEnvs.spot_bezier_env import spotBezierEnv
from spotmicro.util.gui import GUI
from spotmicro.Kinematics.SpotKinematics import SpotModel
from spotmicro.GaitGenerator.Bezier import BezierGait
from spotmicro.OpenLoopSM.SpotOL import BezierStepper

STANCE = 0
SWING = 1


def TransToRp(T):
    """
    Converts a homogeneous transformation matrix into a rotation matrix
    and position vector

    :param T: A homogeneous transformation matrix
    :return R: The corresponding rotation matrix,
    :return p: The corresponding position vector.

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])

    Output:
        (np.array([[1, 0,  0],
                   [0, 0, -1],
                   [0, 1,  0]]),
         np.array([0, 0, 3]))
    """
    T = np.array(T)
    return T[0:3, 0:3], T[0:3, 3]


class GaitState:
    def __init__(self) -> None:
        self.stepLength = 0.0
        self.yawRate = 0
        self.lateralFraction = 0
        self.stepVelocity = 0.001
        self.swingPeriod = 0.2
        self.clearanceHeight = 0.045
        self.penetrationDepth = 0.003
        self.contacts = [False] * 4

        self.targetStepLength = 0

    def updateStepLength(self, dt):
        if self.stepLength < self.targetStepLength:
            self.stepLength += self.targetStepLength * dt


class BodyState:
    def __init__(self) -> None:
        self.position = np.array([0, 0, 0])
        self.rotation = np.array([0, 0, 0])
        self.worldFeetPositions = {}


# class NewBezierGait:
#     def __init__(self) -> None:
#         self.dSref = [0.0, 0.0, 0.5, 0.5]
#         self.dSref = dSref
#         self.Prev_fxyz = [0.0, 0.0, 0.0, 0.0]
#         self.NumControlPoints = 11

#         self.time = 0.0
#         # Touchdown Time
#         self.touchDownTime = 0.0
#         self.time_since_last_TD = 0.0
#         self.StanceSwing = SWING
#         self.SwRef = 0.0
#         self.Stref = 0.0
#         # Whether Reference Foot has Touched Down
#         self.TD = False

#         # Stance Time
#         self.Tswing = 0.2

#         self.ref_idx = 0

#         # Store all leg phases
#         self.Phases = self.dSref

#     def generate_trajectory(self, body_state: BodyState, gait_state: GaitState, dt):
#         if gait_state.stepVelocity != 0:
#             tStance = 2.0 * abs(gait_state.stepLength) / abs(gait_state.stepVelocity)
#         else:
#             tStance = 0.0
#             gait_state.stepLength = 0.0
#             self.TD = False
#             self.time = 0.0
#             self.time_since_last_TD = 0.0

#         gait_state.yawRate *= dt

#         if tStance < dt or tStance > 1.3 * gait_state.swingPeriod:
#             tStance = max(0.0, min(tStance, 1.3 * gait_state.swingPeriod))
#             gait_state.stepLength, gait_state.yawRate = 0.0, 0.0
#             self.TD = False
#             self.time = 0.0
#             self.time_since_last_TD = 0.0

#         if gait_state.contacts[0] and tStance > dt:
#             self.TD = True

#         self.increment(dt, tStance + gait_state.swingPeriod)

#         T_bf = copy.deepcopy(body_state.worldFeetPositions)
#         ref_dS = {"FL": 0.0, "FR": 0.5, "BL": 0.5, "BR": 0.0}
#         for i, (key, Tbf_in) in enumerate(body_state.worldFeetPositions.items()):
#             self.ref_idx = i if key == "FL" else self.ref_idx
#             self.dSref[i] = ref_dS[key]
#             _, p_bf = TransToRp(Tbf_in)
#             step_coord = (
#                 self.get_foot_step(body_state, gait_state, i, key)
#                 if tStance > 0
#                 else np.array([0.0, 0.0, 0.0])
#             )
#             for j in range(3):
#                 T_bf[key][j, 3] += step_coord[j]
#         return T_bf

#     def get_foot_step(self, body_state: BodyState, gait_state: GaitState):
#         phase, stance_swing = self.get_phase(gait_state)
#         self.phases[gait_state.index] = phase + 1.0 if stance_swing == SWING else phase

#         if stance_swing == STANCE:
#             return self.stance_step(phase, gait_state, body_state)
#         return self.swing_step(phase, gait_state, body_state)


class Gait:
    def __init__(
        self,
        env: spotBezierEnv,
        gui: GUI,
        bodyState: BodyState,
        gaitState: GaitState,
        spotModel: SpotModel,
        bezierGait: BezierGait,
    ) -> None:
        self.env = env
        self.gui = gui
        self.bodyState = bodyState
        self.gaitState = gaitState
        self.spot = spotModel
        self.bezierGait = bezierGait

        self.state = self.env.reset()
        self.action = self.env.action_space.sample()
        self.bodyState.worldFeetPositions = copy.deepcopy(self.spot.WorldToFoot)

        self.dt = 0.01

    def step(self):
        self.gaitState.updateStepLength(self.dt)
        self.gui.UserInput(self.bodyState, self.gaitState)
        self.gaitState.contacts = self.state[-4:]
        self.bodyState.worldFeetPositions = copy.deepcopy(self.spot.WorldToFoot)

        self.bodyState.worldFeetPositions = self.bezierGait.GenerateTrajectory(
            self.bodyState, self.gaitState, self.dt
        )

        self.updateEnvironment()

        self.state, _, done, _ = self.env.step(self.action)
        if done:
            print("DONE")
            return True

    def updateEnvironment(self):
        joint_angles = self.spot.IK(
            self.bodyState.rotation,
            self.bodyState.position,
            self.bodyState.worldFeetPositions,
        )
        self.env.pass_joint_angles(joint_angles.reshape(-1))


if __name__ == "__main__":
    env = spotBezierEnv(
        render=True,
        on_rack=False,
        height_field=False,
        draw_foot_path=False,
        env_randomizer=None,
    )
    gui = GUI(env.spot.quadruped)
    bodyState = BodyState()
    gaitState = GaitState()
    spot = SpotModel()
    bezierGait = BezierGait()

    gait = Gait(env, gui, bodyState, gaitState, spot, bezierGait)

    while True:
        done = gait.step()
        if done:
            gait.env.close()
            break
