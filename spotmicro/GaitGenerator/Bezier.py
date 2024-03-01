from enum import Enum
import numpy as np


class Phase(Enum):
    STANCE = 0
    SWING = 1


# Bezier Curves from: https://dspace.mit.edu/handle/1721.1/98270
# Rotation Logic from: http://www.inase.org/library/2014/santorini/bypaper/ROBCIRC/ROBCIRC-54.pdf


def TransToRp(T):
    T = np.array(T)
    return T[0:3, 0:3], T[0:3, 3]


class BezierGait():

    def __init__(self, leg_phases=[0.0, 0.0, 0.5, 0.5], dt=0.01, t_swing=0.2):
        self.leg_phases = leg_phases
        self.prev_foot_pos = np.zeros((4, 3))

        self.num_control_points = 11

        self.dt = dt
        self.time = 0.0
        self.touch_down_time = 0.0
        self.last_touch_down_time = 0.0

        # Trajectory Mode
        self.phase = Phase.SWING

        # Swing Phase value [0, 1] of Reference Foot
        self.sw_ref = 0.0
        self.st_ref = 0.0
        # Whether Reference Foot has Touched Down
        self.touch_down = False

        # Stance Time
        self.t_swing = t_swing

        # Reference Leg
        self.ref_idx = 0

        # Store all leg phases
        self.phases = self.leg_phases

    def reset(self):
        self.prev_foot_pos.fill(0)
        self.time = 0.0
        self.touch_down_time = 0.0
        self.last_touch_down_time = 0.0
        self.phase = Phase.SWING
        self.sw_ref = 0.0
        self.st_ref = 0.0
        self.touch_down = False

    def get_phase(self, index):
        """Retrieves the phase of an individual leg.

        NOTE modification
        from original paper:

        if ti < -t_swing:
           ti += t_stride

        This is to avoid a phase discontinuity if the user selects
        a Step Length and Velocity combination that causes t_stance > t_swing.

        :param index: the leg's index, used to identify the required
                      phase lag
        :param t_stance: the current user-specified stance period
        :param t_swing: the swing period (constant, class member)
        :return: Leg Phase, and StanceSwing (bool) to indicate whether
                 leg is in stance or swing mode
        """
        t_stride = self.t_stance + self.t_swing
        time_index = self.time_index(index, t_stride)

        if time_index < -self.t_swing:
            time_index += t_stride

        is_stance_phase = time_index >= 0.0 and time_index <= self.t_stance
        if is_stance_phase:
            return self.get_stance_phase(time_index, index)

        return self.get_swing_phase(time_index, index)

    def get_stance_phase(self, time_index, index):
        leg_phase = time_index / float(self.t_stance)
        if self.t_stance == 0.0:
            leg_phase = 0.0
        if index == self.ref_idx:
            self.phase = Phase.STANCE
        return leg_phase, Phase.STANCE

    def get_swing_phase(self, time_index, index):
        leg_phase = 0.0
        if time_index >= -self.t_swing and time_index < 0.0:
            leg_phase = min((time_index + self.t_swing) / self.t_swing, 1.0)
        elif time_index > self.t_stance and time_index <= self.t_stride:
            leg_phase = min((time_index - self.t_stance) / self.t_swing, 1.0)
        # Touchdown at End of Swing
        leg_phase = min(leg_phase, 1.0)
        if index == self.ref_idx:
            self.phase = Phase.SWING
            self.sw_ref = leg_phase
            if self.sw_ref >= 0.999:
                self.touch_down = True
        return leg_phase, Phase.SWING

    def time_index(self, index, t_stride):
        """Retrieves the time index for the individual leg

        :param index: the leg's index, used to identify the required
                      phase lag
        :param t_stride: the total leg movement period (t_stance + t_swing)
        :return: the leg's time index
        """
        # NOTE: for some reason python's having numerical issues w this
        # setting to 0 for ref leg by force
        if index == self.ref_idx:
            self.leg_phases[index] = 0.0
        return self.last_touch_down_time - self.leg_phases[index] * t_stride

    def update_clock(self, dt):
        """Increments the Bezier gait generator's internal clock (self.time)

        :param dt: the time step
                      phase lag
        :return: the leg's time index
        """
        self.t_stride = self.t_stance + self.t_swing
        self._check_touch_down()
        self.last_touch_down_time = self.time - self.touch_down_time
        if self.last_touch_down_time > self.t_stride:
            self.last_touch_down_time = self.t_stride
        elif self.last_touch_down_time < 0.0:
            self.last_touch_down_time = 0.0
        self.time += dt

        if self.t_stride < self.t_swing + dt:
            self.time = 0.0
            self.last_touch_down_time = 0.0
            self.touch_down_time = 0.0
            self.sw_ref = 0.0

    def _check_touch_down(self):
        """Checks whether a reference leg touchdown
        has occurred, and whether this warrants
        resetting the touchdown time
        """
        if self.sw_ref >= 0.9 and self.touch_down:
            self.touch_down_time = self.time
            self.touch_down = False
            self.sw_ref = 0.0

    def _bern_stein_poly(self, t, k, point):
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
            * self._binomial(k)
            * np.power(t, k)
            * np.power(1 - t, self.num_control_points - k)
        )

    def _binomial(self, k):
        """Solves the binomial theorem given a Bezier point number
           relative to the total number of Bezier points.

           :param k: Bezier point number
           :returns: Binomial solution
        """
        return np.math.factorial(self.num_control_points) / (
            np.math.factorial(k) * np.math.factorial(self.num_control_points - k)
        )

    def _bezier_swing(self, phase, L, lateral_fraction, clearance_height=0.04):
        """Calculates the step coordinates for the Bezier (swing) period

        :param phase: current trajectory phase
        :param L: step length
        :param lateral_fraction: determines how lateral the movement is
        :param clearance_height: foot clearance height during swing phase

        :returns: X,Y,Z Foot Coordinates relative to unmodified body
        """

        # Bezier Curve Points (12 pts)
        # NOTE: L is HALF of STEP LENGTH
        # Forward Component
        STEP = np.array(
            [
                -L,  # Ctrl Point 0, half of stride len
                -L * 1.4,  # Ctrl Point 1 diff between 1 and 0 = x Lift vel
                -L * 1.5,  # Ctrl Pts 2, 3, 4 are overlapped for
                -L * 1.5,  # Direction change after
                -L * 1.5,  # Follow Through
                0.0,  # Change acceleration during Protraction
                0.0,  # So we include three
                0.0,  # Overlapped Ctrl Pts: 5, 6, 7
                L * 1.5,  # Changing direction for swing-leg retraction
                L * 1.5,  # requires double overlapped Ctrl Pts: 8, 9
                L * 1.4,  # Swing Leg Retraction Velocity = Ctrl 11 - 10
                L,
            ]
        )
        # Account for lateral movements by multiplying with polar coord.
        # lateral_fraction switches leg movements from X over to Y+ or Y-
        # As it tends away from zero
        X = STEP * np.cos(lateral_fraction)

        # Account for lateral movements by multiplying with polar coord.
        # lateral_fraction switches leg movements from X over to Y+ or Y-
        # As it tends away from zero
        Y = STEP * np.sin(lateral_fraction)

        # Vertical Component
        Z = np.array(
            [
                0.0,  # Double Overlapped Ctrl Pts for zero Lift
                0.0,  # Velocity wrt hip (Pts 0 and 1)
                clearance_height * 0.9,  # Triple overlapped control for change in
                clearance_height * 0.9,  # Force direction during transition from
                clearance_height * 0.9,  # follow-through to protraction (2, 3, 4)
                clearance_height * 0.9,  # Double Overlapped Ctrl Pts for Traj
                clearance_height * 0.9,  # Direct Change during Protraction (5, 6)
                clearance_height * 1.1,  # Maximum Clearance at mid Traj, Pt 7
                clearance_height * 1.1,  # Smooth Transition from Protraction
                clearance_height * 1.1,  # To Retraction, Two Ctrl Pts (8, 9)
                0.0,  # Double Overlap Ctrl Pts for 0 Touchdown
                0.0,  # Velocity wrt hip (Pts 10 and 11)
            ]
        )

        stepX = 0.
        stepY = 0.
        stepZ = 0.
        # Bernstein Polynomial sum over control points
        for i in range(len(X)):
            stepX += self._bern_stein_poly(phase, i, X[i])
            stepY += self._bern_stein_poly(phase, i, Y[i])
            stepZ += self._bern_stein_poly(phase, i, Z[i])

        return stepX, stepY, stepZ

    def sine_stance(self, phase, L, lateral_fraction, penetration_depth=0.00):
        """Calculates the step coordinates for the Sinusoidal stance period

        :param phase: current trajectory phase
        :param L: step length
        :param lateral_fraction: determines how lateral the movement is
        :param penetration_depth: foot penetration depth during stance phase

        :returns: X,Y,Z Foot Coordinates relative to unmodified body
        """
        # moves from +L to -L
        step = L * (1.0 - 2.0 * phase)
        stepX = step * np.cos(lateral_fraction)
        stepY = step * np.sin(lateral_fraction)
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
        g_xyz = self.prev_foot_pos[index] - np.array([T_bf[0], T_bf[1], T_bf[2]])

        # Modulate Magnitude to keep tracing circle
        g_mag = np.sqrt((g_xyz[0])**2 + (g_xyz[1])**2)
        th_mod = np.arctan2(g_mag, DefaultBodyToFoot_Magnitude)

        # Angle Traced by Foot for Rotation
        phi_arc = np.pi / 2.0 + th_mod
        phi_arc += DefaultBodyToFoot_Direction * 1 if index == 1 or index == 2 else -1

        return phi_arc

    def swing_step(self, phase, gaitState, T_bf, index):
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
        X_delta_lin, Y_delta_lin, Z_delta_lin = self._bezier_swing(
            phase,
            gaitState.step_length,
            gaitState.lateral_fraction,
            gaitState.clearance_height,
        )

        X_delta_rot, Y_delta_rot, Z_delta_rot = self._bezier_swing(
            phase, gaitState.yaw_rate, phi_arc, gaitState.clearance_height
        )

        coord = np.array([
            X_delta_lin + X_delta_rot, Y_delta_lin + Y_delta_rot,
            Z_delta_lin + Z_delta_rot
        ])

        self.prev_foot_pos[index] = coord

        return coord

    def stance_step(self, phase, gaitState, T_bf, index):
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

        self.prev_foot_pos[index] = coord

        return coord

    def foot_step(self, gaitState, body_foot, index):
        """Calculates the step coordinates in either the Bezier or
        Sine portion of the trajectory depending on the retrieved phase

        :param T_bf: default body-to-foot Vector
        :param index: the foot index in the container

        :returns: Foot Coordinates relative to unmodified body
        """
        leg_phase, foot_phase = self.get_phase(index)
        stored_phase = leg_phase
        if foot_phase == Phase.SWING:
            stored_phase += 1.0

        # Just for keeping track
        self.phases[index] = stored_phase
        if foot_phase == Phase.STANCE:
            return self.stance_step(leg_phase, gaitState, body_foot, index)
        elif foot_phase == Phase.SWING:
            return self.swing_step(leg_phase, gaitState, body_foot, index)

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

        # Catch infeasible timestep
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
