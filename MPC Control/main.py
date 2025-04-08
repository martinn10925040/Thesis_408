import numpy as np
import os
from scipy.signal import cont2discrete
import do_mpc
import matplotlib.pyplot as plt
import casadi as ca

# ------------------------------------------
# 1. Define the continuous-time state-space model
# ------------------------------------------
def create_continuous_state_space(gravity, inertiaX, inertiaY, inertiaZ, mass):
    A_cont = np.array([
        [0,   0,   0,   1,   0,   0,    0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   1,   0,    0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   0,   1,    0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   0,   0,    0,   0,   0,   1,   0,   0],
        [0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   1,   0],
        [0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   1],
        [0, -gravity, 0,  0,   0,   0,    0,   0,   0,   0,   0,   0],
        [gravity,  0,  0,  0,   0,   0,    0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0]
    ])
    
    B_cont = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1/inertiaX, 0, 0],
        [0, 0, 1/inertiaY, 0],
        [0, 0, 0, 1/inertiaZ],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1/mass, 0, 0, 0]
    ])
    
    C_cont = np.eye(12)
    D_cont = np.zeros((12, 4))
    
    return A_cont, B_cont, C_cont, D_cont

# ------------------------------------------
# 2. Discretize the system
# ------------------------------------------
def create_discrete_space(A_cont, B_cont, sample_time):
    sysd = cont2discrete((A_cont, B_cont, np.eye(A_cont.shape[0]), np.zeros((A_cont.shape[0], B_cont.shape[1]))), sample_time)
    A_disc, B_disc = sysd[0], sysd[1]
    return A_disc, B_disc

# ------------------------------------------
# 3. Create a reference trajectory (port of CreateReferencePath)
# ------------------------------------------
def create_reference_path(time_vector, num_steps, x_offset, y_offset, initial_altitude, ideal_landing_time, final_altitude, plot=True):
    half_landing = ideal_landing_time / 2
    final_descent_rate = 0.8  # m/s
    final_segment_height = 1.5  # m
    t_switch = ideal_landing_time - (final_segment_height / final_descent_rate)
    
    c = initial_altitude
    a = (initial_altitude - final_segment_height - final_descent_rate*t_switch) / (t_switch**2)
    b = -final_descent_rate - 2*a*t_switch

    x_ref = np.zeros(num_steps)
    y_ref = np.zeros(num_steps)
    z_ref = np.zeros(num_steps)
    u_ref = np.zeros(num_steps)
    v_ref = np.zeros(num_steps)
    w_ref = np.zeros(num_steps)
    
    for i in range(num_steps):
        t = time_vector[i]
        # X and Y trajectories: converge from (x_offset, y_offset) to zero
        if t <= half_landing:
            x_ref[i] = x_offset * (1 - (t/half_landing)**2)
            y_ref[i] = y_offset * (1 - (t/half_landing)**2)
            u_ref[i] = -2 * x_offset * t / (half_landing**2)
            v_ref[i] = -2 * y_offset * t / (half_landing**2)
        else:
            x_ref[i] = 0
            y_ref[i] = 0
            u_ref[i] = 0
            v_ref[i] = 0

        # Z trajectory: descend from initial_altitude to final_altitude
        if t <= t_switch:
            z_ref[i] = a*t**2 + b*t + c
            w_ref[i] = 2*a*t + b
        elif t <= ideal_landing_time:
            z_ref[i] = final_segment_height - final_descent_rate*(t - t_switch)
            w_ref[i] = -final_descent_rate
        else:
            z_ref[i] = final_altitude
            w_ref[i] = 0

    # Assemble into a 12-state reference trajectory.
    ref_traj = np.zeros((12, num_steps))
    ref_traj[6, :]  = x_ref   # x-position
    ref_traj[7, :]  = y_ref   # y-position
    ref_traj[8, :]  = z_ref   # altitude
    ref_traj[9, :]  = u_ref   # x-velocity
    ref_traj[10, :] = v_ref   # y-velocity
    ref_traj[11, :] = w_ref   # z-velocity
    
    # Optional plotting to visualize the reference path
    # if plot:
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(10,8))
        
    #     plt.subplot(3,1,1)
    #     plt.plot(time_vector, x_ref, label="X Position")
    #     plt.xlabel("Time (s)")
    #     plt.ylabel("X Position (m)")
    #     plt.grid(True)
    #     plt.legend()
        
    #     plt.subplot(3,1,2)
    #     plt.plot(time_vector, y_ref, label="Y Position", color="orange")
    #     plt.xlabel("Time (s)")
    #     plt.ylabel("Y Position (m)")
    #     plt.grid(True)
    #     plt.legend()
        
    #     plt.subplot(3,1,3)
    #     plt.plot(time_vector, z_ref, label="Altitude", color="green")
    #     plt.xlabel("Time (s)")
    #     plt.ylabel("Altitude (m)")
    #     plt.grid(True)
    #     plt.legend()
        
    #     plt.tight_layout()
    #     plt.show()
    
    return ref_traj


# ------------------------------------------
# 4. Set up the MPC controller using do-mpc
# ------------------------------------------
def create_mpc_controller(A_disc, B_disc, sample_time, horizon, weights, input_bounds, state_bounds, ref_traj):
    import casadi as ca
    import do_mpc
    import os
    import numpy as np

    n_states = A_disc.shape[0]
    n_inputs = B_disc.shape[1]

    # Create the model
    model_type = 'discrete'
    model = do_mpc.model.Model(model_type)
    os.system("cls")
    
    # Define state and control variables
    x = model.set_variable(var_type='_x', var_name='x', shape=(n_states, 1))
    u = model.set_variable(var_type='_u', var_name='u', shape=(n_inputs, 1))
    
    # Define TVP variables: 
    # 'r' for the reference (shape (n_states,1)) 
    # and 'alt_lb' for the altitude lower bound (shape (1,1))
    r = model.set_variable(var_type='_tvp', var_name='r', shape=(n_states, 1))
    alt_lb = model.set_variable(var_type='_tvp', var_name='alt_lb', shape=(1, 1))
    
    # Define system dynamics: x_next = A_disc*x + B_disc*u
    x_next = A_disc @ x + B_disc @ u
    model.set_rhs('x', x_next)
    
    # Setup the model
    model.setup()
    
    # Create MPC controller object and add nonlinear constraint
    mpc = do_mpc.controller.MPC(model)
    # Enforce: x[8] - alt_lb >= 0  <=>  -(x[8]-alt_lb) <= 0
    mpc.set_nl_cons('alt_constraint', -(x[8] - alt_lb), 0)
    
    # Set MPC parameters
    mpc.set_param(n_horizon=horizon,
                  t_step=sample_time,
                  state_discretization='discrete',
                  store_full_solution=True)
    
    # Define the TVP function using a manually built DM.
    def tvp_fun(t_now):
        # Obtain the required structure from the optimizer.
        tvp_template = mpc.get_tvp_template()
        
        H = horizon + 1
        # Loop over the horizon and fill the template.
        for k in range(H):
            t_pred = t_now + k * sample_time
            idx = int(t_pred / sample_time)
            if idx >= ref_traj.shape[1]:
                idx = ref_traj.shape[1] - 1
            
            # Create the reference vector from your reference trajectory.
            ref_vec = ca.DM(ref_traj[:, idx].reshape(-1, 1))
            
            # Choose an alternative value based on time.
            if t_pred < 5:
                alt_value = ca.DM(ref_traj[8, idx].reshape(1, 1))
            else:
                alt_value = ca.DM(0)
            
            # Populate the TVP template with the combined values.
            tvp_template['_tvp', k] = ca.vertcat(ref_vec, alt_value)
        
        return tvp_template


    mpc.set_tvp_fun(tvp_fun)
    
    # Define the stage cost to track the reference 'r'
    Q = weights.get('Q', np.eye(n_states))
    R = weights.get('R', np.eye(n_inputs))
    mterm = ca.SX.zeros(1, 1)
    lterm = ca.SX.zeros(1, 1)
    for i in range(n_states):
        lterm += Q[i, i] * (x[i] - r[i])**2
    for j in range(n_inputs):
        lterm += R[j, j] * (u[j])**2

    mpc.set_objective(mterm=mterm, lterm=lterm)
    os.system("cls")
    
    # Set bounds for inputs and states (as before)
    lower_input = np.array([b[0] for b in input_bounds])
    upper_input = np.array([b[1] for b in input_bounds])
    mpc.bounds[('lower', '_u', 'u')] = lower_input
    mpc.bounds[('upper', '_u', 'u')] = upper_input

    lower_state = np.array([b[0] for b in state_bounds])
    upper_state = np.array([b[1] for b in state_bounds])
    mpc.bounds[('lower', '_x', 'x')] = lower_state
    mpc.bounds[('upper', '_x', 'x')] = upper_state

    mpc.setup()
    mpc.set_initial_guess()
    
    return mpc, model











# ------------------------------------------
# 5. Simulation loop
# ------------------------------------------
def flight_simulation(initial_state, A_disc, B_disc, sample_time, num_steps, ref_traj, mpc, model):
    state_history = np.zeros((12, num_steps))
    input_history = np.zeros((4, num_steps))
    current_state = initial_state.copy()
    
    for k in range(num_steps):
        mpc.x0 = current_state
        
        # No manual update of the TVP is needed here,
        # since mpc.set_tvp_fun(tvp_fun) takes care of updating the reference
        
        u0 = mpc.make_step(current_state)
        os.system("cls")
        print('#####################################################################################################' + str(k))
        
        current_state = A_disc @ current_state + B_disc @ u0.reshape(-1, 1)
        if current_state[8, 0] < 0.01:
            current_state[8, 0] = 0
        state_history[:, k] = current_state.flatten()
        input_history[:, k] = u0.flatten()
    
    return state_history, input_history



# ------------------------------------------
# 6. Main script: define parameters, build models, and run simulation
# ------------------------------------------
if __name__ == "__main__":
    
    os.system("cls")
    
    # Simulation parameters
    ideal_landing_time = 7
    total_sim_time = ideal_landing_time + 1
    sample_time = 0.1
    num_steps = int(total_sim_time / sample_time)
    time_vector = np.linspace(0, total_sim_time, num_steps)
    
    # Mission parameters

    x_offset = 2.0
    y_offset = 2.0
    initial_altitude = 5.0
    final_altitude = 0.0
    gravity = 9.81
    
    # UAV parameters
    inertiaX = 0.0165
    inertiaY = 0.0165
    inertiaZ = 0.0165
    mass = 2.0
    
    # Create state-space
    A_cont, B_cont, C_cont, D_cont = create_continuous_state_space(gravity, inertiaX, inertiaY, inertiaZ, mass)
    A_disc, B_disc = create_discrete_space(A_cont, B_cont, sample_time)
    
    # Generate reference trajectory
    ref_traj = create_reference_path(time_vector, num_steps, x_offset, y_offset, initial_altitude, ideal_landing_time-0.425, final_altitude)
    
    # Define weights (adjust these to match your performance criteria)
    weights = {
        'Q': np.diag([0.1, 0.1, 100, 1, 1, 10, 500, 500, 1000, 1200, 900, 600]),
        'R': np.diag([5, 5, 5, 5])  # Example: you might tune these as needed
    }
    
    # Define input constraints (for each of the 4 inputs)
    # Example: thrust limits and torque limits (adjust according to your UAV parameters)
    input_bounds = [
        (-mass * gravity, 2.5 * mass * gravity),  # thrust (deltaT)
        (-6.535, 6.535),                          # roll torque
        (-6.535, 6.535),                          # pitch torque77777777777777
        (-0.36, 0.36)                             # yaw torque
    ]
    
    # Define state bounds for selected states (example bounds for states 7,8,9,10,11,12)
    # Here we set bounds for all 12 states (adjust as needed)
    state_bounds = [(-np.inf, np.inf)] * 12
    state_bounds[6]  = (-0.5, 0.5)      # x position
    state_bounds[7]  = (-0.5, 0.5)      # y position
    state_bounds[8]  = (-0.01, 1.1*initial_altitude)  # altitude z
    state_bounds[9]  = (-1, 1)          # x velocity
    state_bounds[10] = (-1, 1)          # y velocity
    state_bounds[11] = (-2, 0)          # vertical velocity
    
    # Create MPC controller with chosen horizon
    horizon = 40
    mpc, model = create_mpc_controller(A_disc, B_disc, sample_time, horizon, weights, input_bounds, state_bounds, ref_traj)

    # Set initial state (12x1 vector)
    initial_state = np.zeros((12,1))
    initial_state[6] = x_offset       # initial x
    initial_state[7] = y_offset       # initial y
    initial_state[8] = initial_altitude  # initial altitude
    
    # Run simulation
    state_history, input_history = flight_simulation(initial_state, A_disc, B_disc, sample_time, num_steps, ref_traj, mpc, model)
    
    # Plot some results (positions and altitude)
    plt.figure(figsize=(10,6))
    plt.plot(time_vector, state_history[6, :], label='X Position')
    plt.plot(time_vector, state_history[7, :], label='Y Position')
    plt.plot(time_vector, state_history[8, :], label='Altitude')
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.title("UAV Flight Simulation - Target: 5 seconds")
    plt.legend()
    plt.grid(True)
    plt.show()
