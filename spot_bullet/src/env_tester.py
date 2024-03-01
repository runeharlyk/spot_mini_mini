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
