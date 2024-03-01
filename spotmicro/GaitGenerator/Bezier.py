import numpy as np
from spotmicro.Kinematics.LieAlgebra import TransToRp
import copy

STANCE = 0
SWING = 1

# Bezier Curves from: https://dspace.mit.edu/handle/1721.1/98270
# Rotation Logic from: http://www.inase.org/library/2014/santorini/bypaper/ROBCIRC/ROBCIRC-54.pdf


class BezierGait():
    def __init__(self, dSref=[0.0, 0.0, 0.5, 0.5], dt=0.01, Tswing=0.2):
        # Phase Lag Per Leg: FL, FR, BL, BR
        # Reference Leg is FL, always 0
        self.dSref = dSref
        self.Prev_fxyz = [0.0, 0.0, 0.0, 0.0]
        # Number of control points is n + 1 = 11 + 1 = 12
        self.NumControlPoints = 11
        # Timestep
        self.dt = dt

        # Total Elapsed Time
        self.time = 0.0
        # Touchdown Time
        self.TD_time = 0.0
        # Time Since Last Touchdown
        self.time_since_last_TD = 0.0
        # Trajectory Mode
        self.StanceSwing = SWING
        # Swing Phase value [0, 1] of Reference Foot
        self.SwRef = 0.0
        self.Stref = 0.0
        # Whether Reference Foot has Touched Down
        self.touchDown = False

        # Stance Time
        self.Tswing = Tswing

        # Reference Leg
        self.ref_idx = 0

        # Store all leg phases
        self.Phases = self.dSref

    def reset(self):
        """Resets the parameters of the Bezier Gait Generator
        """
        self.Prev_fxyz = [0.0, 0.0, 0.0, 0.0]

        # Total Elapsed Time
        self.time = 0.0
        # Touchdown Time
        self.TD_time = 0.0
        # Time Since Last Touchdown
        self.time_since_last_TD = 0.0
        # Trajectory Mode
        self.StanceSwing = SWING
        # Swing Phase value [0, 1] of Reference Foot
        self.SwRef = 0.0
        self.Stref = 0.0
        # Whether Reference Foot has Touched Down
        self.touchDown = False

    def GetPhase(self, index):
        """Retrieves the phase of an individual leg.

        NOTE modification
        from original paper:

        if ti < -Tswing:
           ti += Tstride

        This is to avoid a phase discontinuity if the user selects
        a Step Length and Velocity combination that causes Tstance > Tswing.

        :param index: the leg's index, used to identify the required
                      phase lag
        :param Tstance: the current user-specified stance period
        :param Tswing: the swing period (constant, class member)
        :return: Leg Phase, and StanceSwing (bool) to indicate whether
                 leg is in stance or swing mode
        """
        StanceSwing = STANCE
        Sw_phase = 0.0
        Tstride = self.Tstance + self.Tswing
        ti = self.Get_ti(index, Tstride)

        # NOTE: PAPER WAS MISSING THIS LOGIC!!
        if ti < -self.Tswing:
            ti += Tstride

        # STANCE
        if ti >= 0.0 and ti <= self.Tstance:
            StanceSwing = STANCE
            if self.Tstance == 0.0:
                Stnphase = 0.0
            else:
                Stnphase = ti / float(self.Tstance)
            if index == self.ref_idx:
                # print("STANCE REF: {}".format(Stnphase))
                self.StanceSwing = StanceSwing
            return Stnphase, StanceSwing
        # SWING
        elif ti >= -self.Tswing and ti < 0.0:
            StanceSwing = SWING
            Sw_phase = (ti + self.Tswing) / self.Tswing
        elif ti > self.Tstance and ti <= Tstride:
            StanceSwing = SWING
            Sw_phase = (ti - self.Tstance) / self.Tswing
        # Touchdown at End of Swing
        if Sw_phase >= 1.0:
            Sw_phase = 1.0
        if index == self.ref_idx:
            # print("SWING REF: {}".format(Sw_phase))
            self.StanceSwing = StanceSwing
            self.SwRef = Sw_phase
            # REF Touchdown at End of Swing
            if self.SwRef >= 0.999:
                self.touchDown = True
            # else:
            #     self.TD = False
        return Sw_phase, StanceSwing

    def Get_ti(self, index, Tstride):
        """Retrieves the time index for the individual leg

        :param index: the leg's index, used to identify the required
                      phase lag
        :param Tstride: the total leg movement period (Tstance + Tswing)
        :return: the leg's time index
        """
        # NOTE: for some reason python's having numerical issues w this
        # setting to 0 for ref leg by force
        if index == self.ref_idx:
            self.dSref[index] = 0.0
        return self.time_since_last_TD - self.dSref[index] * Tstride

    def update_clock(self, dt):
        """Increments the Bezier gait generator's internal clock (self.time)

        :param dt: the time step
                      phase lag
        :param Tstride: the total leg movement period (Tstance + Tswing)
        :return: the leg's time index
        """
        Tstride = self.Tstance + self.Tswing
        self.CheckTouchDown()
        self.time_since_last_TD = self.time - self.TD_time
        if self.time_since_last_TD > Tstride:
            self.time_since_last_TD = Tstride
        elif self.time_since_last_TD < 0.0:
            self.time_since_last_TD = 0.0
        # print("T STRIDE: {}".format(Tstride))
        # Increment Time at the end in case TD just happened
        # So that we get time_since_last_TD = 0.0
        self.time += dt

        # If Tstride = Tswing, Tstance = 0
        # RESET ALL
        if Tstride < self.Tswing + dt:
            self.time = 0.0
            self.time_since_last_TD = 0.0
            self.TD_time = 0.0
            self.SwRef = 0.0

    def CheckTouchDown(self):
        """Checks whether a reference leg touchdown
           has occured, and whether this warrants
           resetting the touchdown time
        """
        if self.SwRef >= 0.9 and self.touchDown:
            self.TD_time = self.time
            self.touchDown = False
            self.SwRef = 0.0

    def BernSteinPoly(self, t, k, point):
        """Calculate the point on the Berinstein Polynomial
           based on phase (0->1), point number (0-11),
           and the value of the control point itself

           :param t: phase
           :param k: point number
           :param point: point value
           :return: Value through Bezier Curve
        """
        return point * self.Binomial(k) * np.power(t, k) * np.power(
            1 - t, self.NumControlPoints - k)

    def Binomial(self, k):
        """Solves the binomial theorem given a Bezier point number
           relative to the total number of Bezier points.

           :param k: Bezier point number
           :returns: Binomial solution
        """
        return np.math.factorial(self.NumControlPoints) / (
            np.math.factorial(k) *
            np.math.factorial(self.NumControlPoints - k))

    def BezierSwing(self, phase, L, LateralFraction, clearance_height=0.04):
        """Calculates the step coordinates for the Bezier (swing) period

           :param phase: current trajectory phase
           :param L: step length
           :param LateralFraction: determines how lateral the movement is
           :param clearance_height: foot clearance height during swing phase

           :returns: X,Y,Z Foot Coordinates relative to unmodified body
        """

        # Polar Leg Coords
        X_POLAR = np.cos(LateralFraction)
        Y_POLAR = np.sin(LateralFraction)

        # Bezier Curve Points (12 pts)
        # NOTE: L is HALF of STEP LENGTH
        # Forward Component
        STEP = np.array([
            -L,  # Ctrl Point 0, half of stride len
            -L * 1.4,  # Ctrl Point 1 diff btwn 1 and 0 = x Lift vel
            -L * 1.5,  # Ctrl Pts 2, 3, 4 are overlapped for
            -L * 1.5,  # Direction change after
            -L * 1.5,  # Follow Through
            0.0,  # Change acceleration during Protraction
            0.0,  # So we include three
            0.0,  # Overlapped Ctrl Pts: 5, 6, 7
            L * 1.5,  # Changing direction for swing-leg retraction
            L * 1.5,  # requires double overlapped Ctrl Pts: 8, 9
            L * 1.4,  # Swing Leg Retraction Velocity = Ctrl 11 - 10
            L
        ])
        # Account for lateral movements by multiplying with polar coord.
        # LateralFraction switches leg movements from X over to Y+ or Y-
        # As it tends away from zero
        X = STEP * X_POLAR

        # Account for lateral movements by multiplying with polar coord.
        # LateralFraction switches leg movements from X over to Y+ or Y-
        # As it tends away from zero
        Y = STEP * Y_POLAR

        # Vertical Component
        Z = np.array([
            0.0,  # Double Overlapped Ctrl Pts for zero Lift
            0.0,  # Veloicty wrt hip (Pts 0 and 1)
            clearance_height * 0.9,  # Triple overlapped control for change in
            clearance_height * 0.9,  # Force direction during transition from
            clearance_height * 0.9,  # follow-through to protraction (2, 3, 4)
            clearance_height * 0.9,  # Double Overlapped Ctrl Pts for Traj
            clearance_height * 0.9,  # Dirctn Change during Protraction (5, 6)
            clearance_height * 1.1,  # Maximum Clearance at mid Traj, Pt 7
            clearance_height * 1.1,  # Smooth Transition from Protraction
            clearance_height * 1.1,  # To Retraction, Two Ctrl Pts (8, 9)
            0.0,  # Double Overlap Ctrl Pts for 0 Touchdown
            0.0,  # Veloicty wrt hip (Pts 10 and 11)
        ])

        stepX = 0.
        stepY = 0.
        stepZ = 0.
        # Bernstein Polynomial sum over control points
        for i in range(len(X)):
            stepX += self.BernSteinPoly(phase, i, X[i])
            stepY += self.BernSteinPoly(phase, i, Y[i])
            stepZ += self.BernSteinPoly(phase, i, Z[i])

        return stepX, stepY, stepZ

    def SineStance(self, phase, L, LateralFraction, penetration_depth=0.00):
        """Calculates the step coordinates for the Sinusoidal stance period

           :param phase: current trajectory phase
           :param L: step length
           :param LateralFraction: determines how lateral the movement is
           :param penetration_depth: foot penetration depth during stance phase

           :returns: X,Y,Z Foot Coordinates relative to unmodified body
        """
        X_POLAR = np.cos(LateralFraction)
        Y_POLAR = np.sin(LateralFraction)
        # moves from +L to -L
        step = L * (1.0 - 2.0 * phase)
        stepX = step * X_POLAR
        stepY = step * Y_POLAR
        if L != 0.0:
            stepZ = -penetration_depth * np.cos(
                (np.pi * (stepX + stepY)) / (2.0 * L))
        else:
            stepZ = 0.0
        return stepX, stepY, stepZ

    def YawCircle(self, T_bf, index):
        """ Calculates the required rotation of the trajectory plane
            for yaw motion

           :param T_bf: default body-to-foot Vector
           :param index: the foot index in the container
           :returns: phi_arc, the plane rotation angle required for yaw motion
        """

        # Foot Magnitude depending on leg type
        DefaultBodyToFoot_Magnitude = np.sqrt(T_bf[0]**2 + T_bf[1]**2)

        # Rotation Angle depending on leg type
        DefaultBodyToFoot_Direction = np.arctan2(T_bf[1], T_bf[0])

        # Previous leg coordinates relative to default coordinates
        g_xyz = self.Prev_fxyz[index] - np.array([T_bf[0], T_bf[1], T_bf[2]])

        # Modulate Magnitude to keep tracing circle
        g_mag = np.sqrt((g_xyz[0])**2 + (g_xyz[1])**2)
        th_mod = np.arctan2(g_mag, DefaultBodyToFoot_Magnitude)

        # Angle Traced by Foot for Rotation
        # FR and BL
        if index == 1 or index == 2:
            phi_arc = np.pi / 2.0 + DefaultBodyToFoot_Direction + th_mod
        # FL and BR
        else:
            phi_arc = np.pi / 2.0 - DefaultBodyToFoot_Direction + th_mod

        return phi_arc

    def SwingStep(self, phase, gaitState, T_bf, index):
        """Calculates the step coordinates for the Bezier (swing) period
           using a combination of forward and rotational step coordinates
           initially decomposed from user input of
           L, LateralFraction and YawRate

           :param phase: current trajectory phase
           :param L: step length
           :param LateralFraction: determines how lateral the movement is
           :param YawRate: the desired body yaw rate
           :param clearance_height: foot clearance height during swing phase
           :param T_bf: default body-to-foot Vector
           :param key: indicates which foot is being processed
           :param index: the foot index in the container

           :returns: Foot Coordinates relative to unmodified body
        """

        # Yaw foot angle for tangent-to-circle motion
        phi_arc = self.YawCircle(T_bf, index)

        # Get Foot Coordinates for Forward Motion
        X_delta_lin, Y_delta_lin, Z_delta_lin = self.BezierSwing(
            phase,
            gaitState.stepLength,
            gaitState.lateralFraction,
            gaitState.clearanceHeight,
        )

        X_delta_rot, Y_delta_rot, Z_delta_rot = self.BezierSwing(
            phase, gaitState.yawRate, phi_arc, gaitState.clearanceHeight
        )

        coord = np.array([
            X_delta_lin + X_delta_rot, Y_delta_lin + Y_delta_rot,
            Z_delta_lin + Z_delta_rot
        ])

        self.Prev_fxyz[index] = coord

        return coord

    def StanceStep(self, phase, gaitState, T_bf, index):
        """Calculates the step coordinates for the Sine (stance) period
           using a combination of forward and rotational step coordinates
           initially decomposed from user input of
           L, LateralFraction and YawRate

           :param phase: current trajectory phase
           :param L: step length
           :param LateralFraction: determines how lateral the movement is
           :param YawRate: the desired body yaw rate
           :param penetration_depth: foot penetration depth during stance phase
           :param T_bf: default body-to-foot Vector
           :param key: indicates which foot is being processed
           :param index: the foot index in the container

           :returns: Foot Coordinates relative to unmodified body
        """

        # Yaw foot angle for tangent-to-circle motion
        phi_arc = self.YawCircle(T_bf, index)

        # Get Foot Coordinates for Forward Motion
        X_delta_lin, Y_delta_lin, Z_delta_lin = self.SineStance(
            phase,
            gaitState.stepLength,
            gaitState.lateralFraction,
            gaitState.penetrationDepth,
        )

        X_delta_rot, Y_delta_rot, Z_delta_rot = self.SineStance(
            phase, gaitState.yawRate, phi_arc, gaitState.penetrationDepth
        )

        coord = np.array([
            X_delta_lin + X_delta_rot, Y_delta_lin + Y_delta_rot,
            Z_delta_lin + Z_delta_rot
        ])

        self.Prev_fxyz[index] = coord

        return coord

    def GetFootStep(self, gaitState, T_bf, index):
        """Calculates the step coordinates in either the Bezier or
           Sine portion of the trajectory depending on the retrieved phase

           :param phase: current trajectory phase
           :param L: step length
           :param LateralFraction: determines how lateral the movement is
           :param YawRate: the desired body yaw rate
           :param clearance_height: foot clearance height during swing phase
           :param penetration_depth: foot penetration depth during stance phase
           :param Tstance: the current user-specified stance period
           :param T_bf: default body-to-foot Vector
           :param index: the foot index in the container
           :param key: indicates which foot is being processed

           :returns: Foot Coordinates relative to unmodified body
        """
        phase, StanceSwing = self.GetPhase(index)
        if StanceSwing == SWING:
            stored_phase = phase + 1.0
        else:
            stored_phase = phase
        # Just for keeping track
        self.Phases[index] = stored_phase
        # print("LEG: {} \t PHASE: {}".format(index, stored_phase))
        if StanceSwing == STANCE:
            return self.StanceStep(phase, gaitState, T_bf, index)
        elif StanceSwing == SWING:
            return self.SwingStep(phase, gaitState, T_bf, index)

    def GenerateTrajectory(self, bodyState, gaitState, dt):
        """Calculates the step coordinates for each foot"""
        gaitState.yawRate *= dt

        self.Tstance = 2.0 * abs(gaitState.stepLength) / abs(gaitState.stepVelocity)
        if gaitState.stepVelocity == 0.0:
            self.Tstance = 0.0
            gaitState.stepLength = 0.0
            self.touchDown = False
            self.time = 0.0
            self.time_since_last_TD = 0.0

        # Catch infeasible timesteps
        if self.Tstance < dt:
            self.Tstance = 0.0
            gaitState.stepLength = 0.0
            self.touchDown = False
            self.time = 0.0
            self.time_since_last_TD = 0.0
            gaitState.yawRate = 0.0
        self.Tstance = min(self.Tstance, 1.3 * self.Tswing)

        if gaitState.contacts[0] == 1 and self.Tstance > dt:
            self.touchDown = True

        self.update_clock(dt)

        T_bf = copy.deepcopy(bodyState.worldFeetPositions)
        ref_dS = {"FL": 0.0, "FR": 0.5, "BL": 0.5, "BR": 0.0}
        for i, (key, Tbf_in) in enumerate(bodyState.worldFeetPositions.items()):
            self.ref_idx = i if key == "FL" else self.ref_idx
            self.dSref[i] = ref_dS[key]
            _, p_bf = TransToRp(Tbf_in)
            if self.Tstance > 0.0:
                step_coord = self.GetFootStep(gaitState, p_bf, i)
            else:
                step_coord = np.array([0.0, 0.0, 0.0])
            for j in range(3):
                T_bf[key][j, 3] += step_coord[j]
        return T_bf
