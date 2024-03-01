from enum import Enum
import numpy as np


class Phase(Enum):
    STANCE = 0
    SWING = 1


# Bezier Curves from: https://dspace.mit.edu/handle/1721.1/98270
# Rotation Logic from: http://www.inase.org/library/2014/santorini/bypaper/ROBCIRC/ROBCIRC-54.pdf


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


class BezierGait():

    def __init__(self, leg_phases=[0.0, 0.0, 0.5, 0.5], dt=0.01, t_swing=0.2):
        self.leg_phases = leg_phases
        self.Prev_fxyz = [0.0, 0.0, 0.0, 0.0]

        self.num_control_points = 11

        self.dt = dt
        self.time = 0.0
        self.touch_down_time = 0.0
        self.last_touch_down_time = 0.0

        # Trajectory Mode
        self.phase = Phase.SWING

        # Swing Phase value [0, 1] of Reference Foot
        self.SwRef = 0.0
        self.Stref = 0.0
        # Whether Reference Foot has Touched Down
        self.touch_down = False

        # Stance Time
        self.t_swing = t_swing

        # Reference Leg
        self.ref_idx = 0

        # Store all leg phases
        self.phases = self.leg_phases

    def reset(self):
        """Resets the parameters of the Bezier Gait Generator
        """
        self.Prev_fxyz = [0.0, 0.0, 0.0, 0.0]
        self.time = 0.0
        self.touch_down_time = 0.0
        self.last_touch_down_time = 0.0
        self.phase = Phase.SWING
        self.SwRef = 0.0
        self.Stref = 0.0
        self.touch_down = False

    def get_phase(self, index):
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
        StanceSwing = Phase.STANCE
        Sw_phase = 0.0
        self.t_stride = self.t_stance + self.t_swing
        ti = self.time_index(index, self.t_stride)

        # NOTE: PAPER WAS MISSING THIS LOGIC!!
        if ti < -self.t_swing:
            ti += self.t_stride

        # STANCE
        if ti >= 0.0 and ti <= self.t_stance:
            Stnphase = ti / float(self.t_stance)
            if self.t_stance == 0.0:
                Stnphase = 0.0
            if index == self.ref_idx:
                self.phase = StanceSwing
            return Stnphase, StanceSwing
        # SWING
        elif ti >= -self.t_swing and ti < 0.0:
            StanceSwing = Phase.SWING
            Sw_phase = (ti + self.t_swing) / self.t_swing
        elif ti > self.t_stance and ti <= self.t_stride:
            StanceSwing = Phase.SWING
            Sw_phase = (ti - self.t_stance) / self.t_swing
        # Touchdown at End of Swing
        Sw_phase = min(Sw_phase, 1.0)
        if index == self.ref_idx:
            # print("SWING REF: {}".format(Sw_phase))
            self.phase = StanceSwing
            self.SwRef = Sw_phase
            # REF Touchdown at End of Swing
            if self.SwRef >= 0.999:
                self.touch_down = True
            # else:
            #     self.TD = False
        return Sw_phase, StanceSwing

    def time_index(self, index, Tstride):
        """Retrieves the time index for the individual leg

        :param index: the leg's index, used to identify the required
                      phase lag
        :param Tstride: the total leg movement period (Tstance + Tswing)
        :return: the leg's time index
        """
        # NOTE: for some reason python's having numerical issues w this
        # setting to 0 for ref leg by force
        if index == self.ref_idx:
            self.leg_phases[index] = 0.0
        return self.last_touch_down_time - self.leg_phases[index] * Tstride

    def update_clock(self, dt):
        """Increments the Bezier gait generator's internal clock (self.time)

        :param dt: the time step
                      phase lag
        :param Tstride: the total leg movement period (Tstance + Tswing)
        :return: the leg's time index
        """
        Tstride = self.t_stance + self.t_swing
        self.CheckTouchDown()
        self.last_touch_down_time = self.time - self.touch_down_time
        if self.last_touch_down_time > Tstride:
            self.last_touch_down_time = Tstride
        elif self.last_touch_down_time < 0.0:
            self.last_touch_down_time = 0.0
        # print("T STRIDE: {}".format(Tstride))
        # Increment Time at the end in case TD just happened
        # So that we get time_since_last_TD = 0.0
        self.time += dt

        # If Tstride = Tswing, Tstance = 0
        # RESET ALL
        if Tstride < self.t_swing + dt:
            self.time = 0.0
            self.last_touch_down_time = 0.0
            self.touch_down_time = 0.0
            self.SwRef = 0.0

    def CheckTouchDown(self):
        """Checks whether a reference leg touchdown
           has occured, and whether this warrants
           resetting the touchdown time
        """
        if self.SwRef >= 0.9 and self.touch_down:
            self.touch_down_time = self.time
            self.touch_down = False
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
        return (
            point
            * self.Binomial(k)
            * np.power(t, k)
            * np.power(1 - t, self.num_control_points - k)
        )

    def Binomial(self, k):
        """Solves the binomial theorem given a Bezier point number
           relative to the total number of Bezier points.

           :param k: Bezier point number
           :returns: Binomial solution
        """
        return np.math.factorial(self.num_control_points) / (
            np.math.factorial(k) * np.math.factorial(self.num_control_points - k)
        )

    def BezierSwing(self, phase, L, lateral_fraction, clearance_height=0.04):
        """Calculates the step coordinates for the Bezier (swing) period

        :param phase: current trajectory phase
        :param L: step length
        :param lateral_fraction: determines how lateral the movement is
        :param clearance_height: foot clearance height during swing phase

        :returns: X,Y,Z Foot Coordinates relative to unmodified body
        """

        # Polar Leg Coords
        X_POLAR = np.cos(lateral_fraction)
        Y_POLAR = np.sin(lateral_fraction)

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
        # lateral_fraction switches leg movements from X over to Y+ or Y-
        # As it tends away from zero
        X = STEP * X_POLAR

        # Account for lateral movements by multiplying with polar coord.
        # lateral_fraction switches leg movements from X over to Y+ or Y-
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

    def sine_stance(self, phase, L, lateral_fraction, penetration_depth=0.00):
        """Calculates the step coordinates for the Sinusoidal stance period

        :param phase: current trajectory phase
        :param L: step length
        :param lateral_fraction: determines how lateral the movement is
        :param penetration_depth: foot penetration depth during stance phase

        :returns: X,Y,Z Foot Coordinates relative to unmodified body
        """
        X_POLAR = np.cos(lateral_fraction)
        Y_POLAR = np.sin(lateral_fraction)
        # moves from +L to -L
        step = L * (1.0 - 2.0 * phase)
        stepX = step * X_POLAR
        stepY = step * Y_POLAR
        stepZ = 0.0
        if L != 0.0:
            stepZ = -penetration_depth * np.cos((np.pi * (stepX + stepY)) / (2.0 * L))

        return stepX, stepY, stepZ

    def yaw_circle(self, T_bf, index):
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
        phi_arc = np.pi / 2.0 + th_mod
        phi_arc += DefaultBodyToFoot_Direction * 1 if index == 1 or index == 2 else -1

        return phi_arc

    def SwingStep(self, phase, gaitState, T_bf, index):
        """Calculates the step coordinates for the Bezier (swing) period
        using a combination of forward and rotational step coordinates
        initially decomposed from user input of
        L, lateral_fraction and yaw_rate

        :param phase: current trajectory phase
        :param L: step length
        :param lateral_fraction: determines how lateral the movement is
        :param yaw_rate: the desired body yaw rate
        :param clearance_height: foot clearance height during swing phase
        :param T_bf: default body-to-foot Vector
        :param key: indicates which foot is being processed
        :param index: the foot index in the container

        :returns: Foot Coordinates relative to unmodified body
        """

        # Yaw foot angle for tangent-to-circle motion
        phi_arc = self.yaw_circle(T_bf, index)

        # Get Foot Coordinates for Forward Motion
        X_delta_lin, Y_delta_lin, Z_delta_lin = self.BezierSwing(
            phase,
            gaitState.step_length,
            gaitState.lateral_fraction,
            gaitState.clearance_height,
        )

        X_delta_rot, Y_delta_rot, Z_delta_rot = self.BezierSwing(
            phase, gaitState.yaw_rate, phi_arc, gaitState.clearance_height
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
        L, lateral_fraction and yaw_rate

        :param phase: current trajectory phase
        :param gaitState: current gait state
        :param T_bf: default body-to-foot Vector
        :param index: the foot index in the container

        :returns: Foot Coordinates relative to unmodified body
        """

        # Yaw foot angle for tangent-to-circle motion
        phi_arc = self.yaw_circle(T_bf, index)

        # Get Foot Coordinates for Forward Motion
        X_delta_lin, Y_delta_lin, Z_delta_lin = self.sine_stance(
            phase,
            gaitState.step_length,
            gaitState.lateral_fraction,
            gaitState.penetration_depth,
        )

        X_delta_rot, Y_delta_rot, Z_delta_rot = self.sine_stance(
            phase, gaitState.yaw_rate, phi_arc, gaitState.penetration_depth
        )

        coord = np.array([
            X_delta_lin + X_delta_rot, Y_delta_lin + Y_delta_rot,
            Z_delta_lin + Z_delta_rot
        ])

        self.Prev_fxyz[index] = coord

        return coord

    def foot_step(self, gaitState, body_foot, index):
        """Calculates the step coordinates in either the Bezier or
        Sine portion of the trajectory depending on the retrieved phase

        :param T_bf: default body-to-foot Vector
        :param index: the foot index in the container

        :returns: Foot Coordinates relative to unmodified body
        """
        phase, StanceSwing = self.get_phase(index)
        stored_phase = phase
        if StanceSwing == Phase.SWING:
            stored_phase += 1.0

        # Just for keeping track
        self.phases[index] = stored_phase
        if StanceSwing == Phase.STANCE:
            return self.StanceStep(phase, gaitState, body_foot, index)
        elif StanceSwing == Phase.SWING:
            return self.SwingStep(phase, gaitState, body_foot, index)

    def generate_trajectory(self, bodyState, gaitState, dt):
        """Calculates the step coordinates for each foot"""
        gaitState.yaw_rate *= dt

        self.t_stance = 2.0 * abs(gaitState.step_length) / abs(gaitState.step_velocity)
        if gaitState.step_velocity == 0.0:
            self.t_stance = 0.0
            gaitState.step_length = 0.0
            self.touch_down = False
            self.time = 0.0
            self.last_touch_down_time = 0.0

        # Catch infeasible timesteps
        if self.t_stance < dt:
            self.t_stance = 0.0
            gaitState.step_length = 0.0
            self.touch_down = False
            self.time = 0.0
            self.last_touch_down_time = 0.0
            gaitState.yaw_rate = 0.0
        self.t_stance = min(self.t_stance, 1.3 * self.t_swing)

        if gaitState.contacts[0] == 1 and self.t_stance > dt:
            self.touch_down = True

        self.update_clock(dt)

        ref_dS = {"FL": 0.0, "FR": 0.5, "BL": 0.5, "BR": 0.0}
        for i, (key, Tbf_in) in enumerate(bodyState.worldFeetPositions.items()):
            self.ref_idx = i if key == "FL" else self.ref_idx
            self.leg_phases[i] = ref_dS[key]
            _, leg_feet_positions = TransToRp(Tbf_in)
            step_coord = (
                self.foot_step(gaitState, leg_feet_positions, i)
                if self.t_stance > 0.0
                else np.array([0.0, 0.0, 0.0])
            )
            for j in range(3):
                bodyState.worldFeetPositions[key][j, 3] += step_coord[j]
