import numpy as np
import gym
from gym import spaces

# OrbitalEnvironment simulates a 2D gravitational orbital system.
# Takes the gravitational constant (GM), initial radius (r0), initial velocity (v0), time step (dt),
# maximum simulation steps, and an optional reward function.
# Outputs the current state after each step (x, y, vx, vy) and reward.
class OrbitalEnvironment:
    def __init__(self, GM=1.0, r0=None, v0=1.0, dt=0.01, max_steps=5000, reward_function=None):
        """
        Args:
            GM: Gravitational constant (float).
            r0: Initial orbital radius (float). If None, a random value is generated.
            v0: Initial velocity (float).
            dt: Time step for the simulation (float).
            max_steps: Maximum number of simulation steps (int).
            reward_function: Optional function for calculating rewards. Defaults to exponential radial difference.

        Returns:
            None. Initializes the orbital environment state.
        """
        self.GM = GM
        self.dt = dt
        self.init_r = r0 if r0 is not None else np.random.uniform(0.2, 4.0)
        self.enforce_r = True if r0 is not None else False
        self.x = self.init_r
        self.y = 0.0
        self.vx = 0.0
        self.vy = np.sqrt(self.GM / self.init_r)
        self.max_steps = max_steps
        self.current_step = 0
        self.reward_function = reward_function or self.default_reward
        self.reset()

    def reset(self):
        """
        Resets the environment to the initial state.
        Returns: Initial state as a numpy array (x, y, vx, vy).
        """
        self.x = self.init_r if self.enforce_r else np.random.uniform(0.2, 4.0)
        self.y = 0.0
        self.vx = 0.0
        self.vy = np.sqrt(self.GM / self.init_r)
        self.current_step = 0
        state = np.array([self.x, self.y, self.vx, self.vy])
        return state
    
    def step(self, action):
        """
        Advances the environment state by one timestep using Runge-Kutta (RK4) integration.
        Args:
            action: Tangential thrust value (float).

        Returns:
            Tuple of (new state, reward, done):
            - new state: Updated state (x, y, vx, vy) as a numpy array.
            - reward: Calculated reward based on the current state and action.
            - done: Boolean indicating whether the simulation is complete.
        """
        # ------- OVERVIEW OF FUNCTION (since it is slightly complicated)
        # We need to find the acceleration, and apply that acceleration to tells us what the position vector [x, y] and
        # velocity vector [vx, vy] of the next 'state' will be.
        # The accleration of the ship that will determine the this next 'state' is a combination of all the forces in the scene.
        # F = ma. If you think about it simply, there is force applied by the gravitational pull of the center mass, and the force applied by the craft's engine.
        # so a_total = a_gravity + a_craft. We will compute a_total, and use v_f = v_0 + at and x_f = x_0 + (v_x0 t) + (0.5 a_x0 t) to get the updated state's velocity + position.


        # A quick note on precise definition of thrust for non-mech-e students like myself: 
        # Thrust is a mechanical force that moves the aircraft through the air. 
        # It is generated most often through the reaction of accelerating a mass of gas. 
        # The engine does work on the gas and as the gas is accelerated to the rear, the engine is accelerated in the opposite direction. 


        # ----- TODO: Compute a_total = a_gravity + a_craft. 
        # Apply a_total using Runge-Kuta 4 Midpoints Integrator Formula to compute the next self.x, self.y, self.vx,  self.vy.
        # ----- 1. Compute a_gravity (acceleration due to gravitty of central mass) ----
        def acceleration(state):
            """
            Helper function to compute the gravitational acceleration. Returns acceleration pulling craft IN toward center mass.

            The acceleration is given by Newton's law of gravitation:
                a = -GM / r^2 * r̂
            where r̂ is the radial unit vector from the origin to the spacecraft.
            The result always points radially inward toward the central mass.
        
            Args:
                state (numpy.ndarray): Current state of the system as an array 
                    [x, y, vx, vy]. Only x and y are used here.
        
            Returns:
                numpy.ndarray: A 2D vector [ax, ay] representing gravitational
                acceleration in Cartesian coordinates (x, y).
            """
            # Extract x and y position
            x, y = state[:2]

            # Distance from central mass (origin)
            dist = np.sqrt(x**2 + y**2)

            # Prevent division-by-zero (clip very small or very large distances)
            dist = np.clip(dist, 1e-5, 5.0)

            # Radial unit vector pointing outward
            rhat = np.array([x, y]) / dist

            # Inward gravitational acceleration vector
            return -self.GM / (dist**2) * rhat

        # ----- 2. Package the current state vector -----
        # State format: [x, y, vx, vy]
        state = np.array([self.x, self.y, self.vx, self.vy])

        # ----- 3. RK4 integration steps -----
        # k1: acceleration and velocity slope at the beginning of the interval
        k1_v = self.dt * acceleration(state)             # acceleration * dt
        k1_p = self.dt * np.array([self.vx, self.vy])    # velocity * dt
        
        # k2: slope at midpoint using k1
        state_mid = state + 0.5 * np.concatenate([k1_p, k1_v])
        k2_v = self.dt * acceleration(state_mid)         # acceleration at midpoint
        k2_p = self.dt * np.array([self.vx + 0.5 * k1_v[0],
                                   self.vy + 0.5 * k1_v[1]])
        
        # k3: slope at midpoint using k2
        state_mid = state + 0.5 * np.concatenate([k2_p, k2_v])
        k3_v = self.dt * acceleration(state_mid)         # acceleration at midpoint
        k3_p = self.dt * np.array([self.vx + 0.5 * k2_v[0],
                                   self.vy + 0.5 * k2_v[1]])
        
        # k4: slope at end of interval using k3
        state_end = state + np.concatenate([k3_p, k3_v])
        k4_v = self.dt * acceleration(state_end)         # acceleration at end
        k4_p = self.dt * np.array([self.vx + k3_v[0],
                                   self.vy + k3_v[1]])

        # ----- 4. Combine RK4 results into weighted average to 
        # get new x,y,vx,vy after applying a_gravity -----
        # RK4 uses a weighted sum: (k1 + 2*k2 + 2*k3 + k4) / 6
        # This gives a more accurate update than Euler's method.
        self.vx += (k1_v[0] + 2*k2_v[0] + 2*k3_v[0] + k4_v[0]) / 6
        self.vy += (k1_v[1] + 2*k2_v[1] + 2*k3_v[1] + k4_v[1]) / 6
        self.x  += (k1_p[0] + 2*k2_p[0] + 2*k3_p[0] + k4_p[0]) / 6
        self.y  += (k1_p[1] + 2*k2_p[1] + 2*k3_p[1] + k4_p[1]) / 6

        # ----- 5. Compute a_craft (acceleration due to craft's thrust)
        action = np.array([0, action[0]])  # Tangential thrust only. Action is defined in local (radial, tangential). 

        # ----- 6. Rotate action's thrust vector from local (origin is the craft) into global (origin is central mass)coordinates -----
        # Radial distance (vector ponting from central mass ---> to craft)
        dist = np.sqrt(self.x**2 + self.y**2)
        dist = max(dist, 1e-5)  

        # Radial unit vector
        rhat = np.array([self.x, self.y]) / dist

        # Build the rotation matrix not from a fixed θ but from the current position of the spacecraft:
        # First column = radial vector (0, 1)
        # Second column = tangential vector (-1, 0)
        rotation_matrix = np.array([[rhat[0], -rhat[1]], [rhat[1], rhat[0]]])

        # Project the local thrust (action) onto the global x,y coordinates using the rotation matrix
        thrust = rotation_matrix @ action
        # thrust = [action_x_global, action_y_global]

        # ----- 6. Computer v_f = v_0 + a(t)
        self.vx += thrust[0] * self.dt
        self.vy += thrust[1] * self.dt
        # ----- 7. No need to updae self.x and self.y here because 
        # RK4 already advanced positions x,y forward in time, using the old velocity. 
        # Thrust alters velocity after this integration step, so the effect on position 
        # will show up during the next call to step().

        # Update state
        state = np.array([self.x, self.y, self.vx, self.vy])

        # Update reward
        reward = self.reward_function(action[1])

        # Check if the episode is done
        done = dist > 5.0 or dist < 0.1 or self.current_step >= self.max_steps
        self.current_step += 1

        return state, reward, done

    def default_reward(self, action):
        """
        Default reward function based on radial distance from the target orbit and action penalty.
        Args:
            action: Tangential thrust value (float).
        Returns:
            Reward value (float).
        """
        # Current orbit radius is distance of craft from origin
        r = np.sqrt(self.x**2 + self.y**2)
        # Target orbit is radius = 1.0. r_err is how far off we are from this ideal orbit.
        r_err = r - 1.0

        # abs(self.init_r - 1) is your initial error from the target orbit.
        r_max_err = max(abs(self.init_r - 1), 1e-2)
        # scale the error relative to init error, and then bound it to be between -2 and 2
        scaled_r_err = np.clip((r_err / r_max_err) * 2, -2, 2)

        #  Reward is highest (≈1.0) when scaled_r_err = 0 (perfect orbit). 
        # It decreases smoothly as error grows, shaped like a Gaussian curve exp(-x^2).
        reward = np.exp(-scaled_r_err**2)

        # Reward is reduced if the agent fires its thrusters strongly.
        # Small actions (close to 0) give penalty ≈ 1, large actions shrink it toward 0.
        action_penalty = np.exp(-action**2)
        return reward * action_penalty

class OrbitalEnvWrapper(gym.Env):
    def __init__(self, r0=None, reward_function=None):
        """
        Args:
            r0: Initial orbital radius (float). If None, a random value is generated.
            reward_function: Optional custom reward function.
        
        Returns:
            None. Initializes the environment and action/observation spaces.
        """
        super(OrbitalEnvWrapper, self).__init__()
        self.env = OrbitalEnvironment(r0=r0, reward_function=reward_function)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.state = None
        self.episode_data = []
        self.prev_r_err = None
        self.integral_r_err = 0.0

    def reset(self):
        """
        Resets the environment to the initial state and clears episode data.
        Returns: Initial observation (numpy array) after conversion by _convert_state().
        """
        self.episode_data = []
        self.state = self.env.reset()
        self.prev_r_err = None
        self.integral_r_err = 0.0
        return self._convert_state(self.state, 0.0, 0.0)

    def step(self, action):
        """
        Performs one step in the environment using the provided action.
        Args:
            action: Tangential thrust action (float).

        Returns:
            Tuple of (observation, reward, done, info):
            - observation: Next state (numpy array).
            - reward: Reward from the current step (float).
            - done: Whether the episode has finished (boolean).
            - info: Additional info dictionary containing the state.
        """
        self.state, base_reward, done = self.env.step(action)

        # Extract state variables
        x, y, vx, vy = self.state[0], self.state[1], self.state[2], self.state[3]
        r = np.sqrt(x**2 + y**2)
        t = self.env.current_step * self.env.dt  # Current time

        # Constants for Hohmann transfer
        r_initial = self.env.init_r
        r_final = 1.0
        transfer_time = np.pi * np.sqrt(((r_initial + r_final) / 2)**3 / self.env.GM)

        # Compute expected radius at current time
        if t < transfer_time:
            # On elliptical transfer orbit
            a_transfer = (r_initial + r_final) / 2
            e_transfer = (r_final - r_initial) / (r_final + r_initial)
            # Mean motion
            n_transfer = np.sqrt(self.env.GM / a_transfer**3)
            # Mean anomaly
            M = n_transfer * t
            # Eccentric anomaly approximation
            E = M  # For small eccentricities
            # True anomaly
        
            theta = 2 * np.arctan2(np.sqrt(1 + e_transfer) * np.sin(E / 2),
                                np.sqrt(1 - e_transfer) * np.cos(E / 2))
            # Expected radius
            r_expected = a_transfer * (1 - e_transfer**2) / (1 + e_transfer * np.cos(theta))
        else:
            # Circularize at r_final
            r_expected = r_final

        # Compute errors
        r_err = r - r_expected

        d_r_err = (r_err - self.prev_r_err) / self.env.dt if self.prev_r_err is not None else 0.0
        self.prev_r_err = r_err

        self.integral_r_err += r_err * self.env.dt

        max_timesteps_passed = self.env.max_steps * self.env.dt

        # Normalize errors
        r_err_norm = r_err / r_expected
        d_r_err_norm = d_r_err / (r_expected / transfer_time)
        int_r_err_norm = self.integral_r_err / (r_expected * max_timesteps_passed)

        # Time-dependent penalty factor
        
        time_factor = t / max_timesteps_passed

        # Penalty factors using exponential decay
        k1, k2, k3 = 1.0, 1.0, 1.0  # Tunable constants
        penalty_r_err = np.exp(-k1 * abs(r_err_norm) * (1 + time_factor))
        penalty_d_r_err = np.exp(-k2 * abs(d_r_err_norm) * (1 + time_factor))
        penalty_int_r_err = np.exp(-k3 * abs(int_r_err_norm) * (1 + time_factor))

        # Compute reward
        base_reward = 1.0  # Maximum possible reward per step
        reward = base_reward * penalty_r_err * penalty_d_r_err * penalty_int_r_err

        # Ensure reward is not too small
        min_reward = 0.01  # Minimal reward to avoid zero
        reward = max(reward, min_reward)

        # Heavy penalty for divergence
        if r > 2 * r_final or r < r_initial / 2:
            reward = min_reward  # Set reward to minimal value

        # Save episode data
        self.episode_data.append([
            x, y, vx, vy, reward, action[0], r_err_norm, d_r_err_norm, int_r_err_norm
        ])

        # Info dictionary
        info = {
            "state": (x, y, vx, vy),
            "r_err_norm": r_err_norm,
            "d_r_err_norm": d_r_err_norm,
            "int_r_err_norm": int_r_err_norm
        }

        # Prepare next observation
        observation = self._convert_state(self.state, d_r_err_norm, int_r_err_norm)

        return observation, reward, done, info

    def _convert_state(self, state, d_r_err_norm, int_r_err_norm):
        """
        Converts the raw state into a processed observation for the RL model.
        Args:
            state: Raw state (numpy array [x, y, vx, vy]).
            d_r_err_norm: Derivative of radial error.
            int_r_err_norm: Integral of radial error.

        Returns:
            Processed observation (numpy array [scaled_r_err, v_radial, v_tangential, initial_r, timestep, flag,
                                                specific_energy, angular_momentum, d_r_err, scaled_integral_r_err]).
        """
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2)
        r = max(r, 1e-5)  # Avoid division by zero
        v_radial = (x * vx + y * vy) / r
        v_tangential = (x * vy - y * vx) / r
        initial_r = self.env.init_r
        flag = 1.0 if np.abs(r - 1.0) < 0.01 else 0.0
        specific_energy = 0.5 * (vx**2 + vy**2) - self.env.GM / r
        angular_momentum = r * v_tangential

        # Scaling for radial error and integral of radial error
        r_err = r - 1.0
        r_max_err = max(abs(self.env.init_r - 1), 1e-2)
        scaled_r_err = np.clip((r_err / r_max_err) * 2, -2, 2)

        # Pack processed state
        state = np.array([
            scaled_r_err,         # Scaled radius error (=1 when at init_r)
            v_radial,             # Radial velocity
            v_tangential,         # Tangential velocity
            1 - initial_r,        # Initial radial error
            flag,                 # Flag if at one of the Hohmann thrust points
            specific_energy,      # KE + PE
            angular_momentum,     # Rotational momentum
            # d_r_err_norm,         # Change in radial error
            # int_r_err_norm        # Scaled cumulative radial error
        ], dtype=np.float32)

        return state


class MultiShipOrbitalEnvironment:
    """
    Manages multiple ships in a shared orbital environment, where each ship can be controlled independently
    and all ships interact gravitationally.
    """
    def __init__(self, GM=1.0, dt=0.01, max_steps=None):  # Changed max_steps to None by default
        self.GM = GM
        self.dt = dt
        self.max_steps = max_steps  # None means no step limit
        self.ships = {}  # ship_id: state dict
        self.current_step = 0

    def add_ship(self, ship_id, r0=None, v0=1.0, control_type='ai'):
        # Each ship has its own state, similar to OrbitalEnvironment
        if r0 is None:
            r0 = np.random.uniform(0.2, 4.0)
            
        x = r0
        y = 0.0
        vx = 0.0
        vy = np.sqrt(self.GM / r0)
        self.ships[ship_id] = {
            'x': x, 'y': y, 'vx': vx, 'vy': vy, 'init_r': r0, 'done': False,
            'control_type': control_type,  # 'ai' or 'manual'
            'heading': 0.0,  # Current heading angle (radians from the x-axis) keep adding turn_rate radians to this
            'thrust': 0.0,   # Current thrust magnitude
            'turn_rate': 0.0, # Current turn rate
            'steps': 0  # Track individual ship steps
        }

    def remove_ship(self, ship_id):
        if ship_id in self.ships:
            del self.ships[ship_id]
    
    def reset(self):
        self.ships.clear()
        self.current_step = 0

    def step(self, actions):
        """
        actions: dict of {ship_id: action}
        For AI ships: action is float (tangential thrust)
        For manual ships: action is dict {'turn': float, 'thrust': float}
        Steps all ships forward, applying their actions and mutual gravity.
        """
        # Gather all positions for mutual gravity
        positions = {sid: (s['x'], s['y']) for sid, s in self.ships.items() if not s['done']}
        
        for ship_id, ship in self.ships.items():
            if ship['done']:
                continue
                
            # Increment ship's step counter
            ship['steps'] += 1
                
            # Handle different control types
            if ship['control_type'] == 'ai':
                # AI control: tangential thrust only
                action = actions.get(ship_id, 0.0)
                self._apply_ai_control(ship, action)

            elif ship['control_type'] == 'manual':
                # Manual control: direction and thrust
                action = actions.get(ship_id, {'turn': 0.0, 'thrust': 0.0})
                self._apply_manual_control(ship, action)
            
            # Apply physics (gravity + thrust)
            self._apply_physics(ship, positions)
            
            # Check if ship is done - only based on bounds, not step limit
            dist = np.sqrt(ship['x']**2 + ship['y']**2)
            if dist > 5.0 or dist < 0.1:  # Removed max_steps check
                ship['done'] = True
                
        self.current_step += 1

    def _apply_ai_control(self, ship, action):
        """Apply AI control (tangential thrust only)"""
        # Store the tangential thrust for physics step
        ship['tangential_thrust'] = action

    def _apply_manual_control(self, ship, action):
        """Apply manual control (direction and thrust)"""
        # Update heading based on turn rate
        ship['turn_rate'] = action.get('turn', 0.0)
        ship['heading'] += ship['turn_rate'] * self.dt
        
        # Update thrust
        ship['thrust'] = action.get('thrust', 0.0)
        
        # Ensure heading is properly initialized if not present
        if 'heading' not in ship:
            ship['heading'] = 0.0

    def _apply_physics(self, ship, positions):
        """Apply physics to a ship (gravity + thrust)"""
        x, y = ship['x'], ship['y']
        vx, vy = ship['vx'], ship['vy']
        
        # Compute gravitational acceleration from central mass
        dist = np.sqrt(x**2 + y**2)
        dist = np.clip(dist, 1e-5, 5.0)
        rhat = np.array([x, y]) / dist
        acc = -self.GM / (dist**2) * rhat
        
        # Add gravity from other ships
        for other_id, (ox, oy) in positions.items():
            if other_id == ship['id'] if hasattr(ship, 'id') else False:
                continue
            dx, dy = ox - x, oy - y
            # distance between ships
            odist = np.sqrt(dx**2 + dy**2)
            if odist < 1e-3:
                continue  # skip self or near-collisions

            # unit vector pointing from ship to other ship
            o_rhat = np.array([dx, dy]) / odist
            # gravity formula: F = G * m1 * m2 / r^2
            acc += -self.GM / (odist**2) * o_rhat * 0.1  # scale down ship-ship gravity
        
        # Apply thrust based on control type
        if ship['control_type'] == 'ai':
            # AI: tangential thrust
            if 'tangential_thrust' in ship:
                # Apply tangential thrust
                rotation_matrix = np.array([[rhat[0], -rhat[1]], [rhat[1], rhat[0]]])
                thrust = rotation_matrix @ np.array([0, ship['tangential_thrust']])
                acc += thrust
        
        elif ship['control_type'] == 'manual':
            # Manual: thrust in current heading direction
            if ship['thrust'] > 0:
                thrust_direction = np.array([np.cos(ship['heading']), np.sin(ship['heading'])])
                thrust = thrust_direction * ship['thrust']
                acc += thrust
        
        # RK4 integration
        # k1
        k1_v = self.dt * acc
        k1_p = self.dt * np.array([vx, vy])
        # k2
        k2_v = self.dt * acc  # For simplicity, use same acc
        k2_p = self.dt * np.array([vx + 0.5 * k1_v[0], vy + 0.5 * k1_v[1]])
        # k3
        k3_v = self.dt * acc
        k3_p = self.dt * np.array([vx + 0.5 * k2_v[0], vy + 0.5 * k2_v[1]])
        # k4
        k4_v = self.dt * acc
        k4_p = self.dt * np.array([vx + k3_v[0], vy + k3_v[1]])
        
        # Update velocity and position
        vx += (k1_v[0] + 2 * k2_v[0] + 2 * k3_v[0] + k4_v[0]) / 6
        vy += (k1_v[1] + 2 * k2_v[1] + 2 * k3_v[1] + k4_v[1]) / 6
        x += (k1_p[0] + 2 * k2_p[0] + 2 * k3_p[0] + k4_p[0]) / 6
        y += (k1_p[1] + 2 * k2_p[1] + 2 * k3_p[1] + k4_p[1]) / 6
        
        # Update ship state
        ship['x'], ship['y'], ship['vx'], ship['vy'] = x, y, vx, vy

    def get_states(self):
        """
        Returns a dict of {ship_id: {'x':..., 'y':..., 'vx':..., 'vy':..., 'done':..., 'heading':..., 'thrust':...}}
        """
        return {sid: dict(s) for sid, s in self.ships.items()}
