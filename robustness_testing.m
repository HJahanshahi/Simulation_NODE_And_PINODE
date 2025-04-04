function robustness_testing()
    % System parameters
    params = struct();
    params.m0 = 60; params.m1 = 6; params.m2 = 5;
    params.I0 = 24; params.I1 = 1; params.I2 = 0.8;
    params.l0 = 0.75; params.l1 = 0.75; params.l2 = 0.75;
    params.lc1 = params.l1/2; params.lc2 = params.l2/2;

    % Simulation parameters
    T = 10;
    dt = 0.01;
    t = 0:dt:T;
    
    % Load the trained models
    physics_model = load('physics_informed_model.mat').net_params;
    vanilla_model = load('vanilla_model.mat').net_params;
    
    % Run the robustness tests
    fprintf('Running robustness tests...\n');
    
    % 1. Sensor noise test
    fprintf('\n1. Testing robustness to sensor noise...\n');
    test_sensor_noise(physics_model, vanilla_model, t, params);
    
    % 2. Parameter uncertainty test
    fprintf('\n2. Testing robustness to parameter uncertainty...\n');
    test_parameter_uncertainty(physics_model, vanilla_model, t, params);
    
    % 3. External disturbance test
    fprintf('\n3. Testing robustness to external disturbances...\n');
    test_external_disturbances(physics_model, vanilla_model, t, params);
    
    % 4. Different initial conditions test
    fprintf('\n4. Testing robustness to extreme initial conditions...\n');
    test_extreme_initial_conditions(physics_model, vanilla_model, t, params);
    
    fprintf('\nRobustness testing complete. Results saved to robustness_results.mat\n');
end

function test_sensor_noise(physics_model, vanilla_model, t, params)
    % Generate test trajectories
    num_test = 20;
    [x0_test, xTest] = generate_trajectories(num_test, t, params);
    
    % Noise levels to test
    noise_levels = [0.001, 0.005, 0.01, 0.05, 0.1];
    physics_errors = zeros(length(noise_levels), num_test);
    vanilla_errors = zeros(length(noise_levels), num_test);
    
    for n = 1:length(noise_levels)
        noise_level = noise_levels(n);
        fprintf('  Testing noise level: %.3f\n', noise_level);
        
        for i = 1:num_test
            % Get true trajectory
            x_true = xTest{i};
            
            % Add noise to initial state
            x0_noisy = x0_test(:,i) + noise_level * randn(size(x0_test(:,i)));
            
            % Predict trajectories
            x_physics = predict_trajectory(t, x0_noisy, physics_model);
            x_vanilla = predict_vanilla_trajectory(t, x0_noisy, vanilla_model);
            
            % Calculate errors (normalized by the magnitude of the state vector)
            physics_errors(n,i) = mean(vecnorm(x_physics - x_true, 2, 1) ./ vecnorm(x_true, 2, 1));
            vanilla_errors(n,i) = mean(vecnorm(x_vanilla - x_true, 2, 1) ./ vecnorm(x_true, 2, 1));
        end
    end
    
    % Calculate average errors across test cases
    avg_physics_errors = mean(physics_errors, 2);
    avg_vanilla_errors = mean(vanilla_errors, 2);
    
    % Plot results
    figure('Name', 'Sensor Noise Robustness');
    semilogx(noise_levels, avg_physics_errors, 'b-o', ...
             noise_levels, avg_vanilla_errors, 'r-o', 'LineWidth', 1.5);
    title('Robustness to Initial State Noise');
    xlabel('Noise Level (std dev)');
    ylabel('Normalized Trajectory Error');
    legend('Physics-Informed', 'Vanilla NN', 'Location', 'northwest');
    grid on;
    saveas(gcf, 'noise_robustness.png');
    
    % Calculate improvement
    improvement = (avg_vanilla_errors - avg_physics_errors) ./ avg_vanilla_errors * 100;
    
    % Print results
    fprintf('\n  Sensor Noise Robustness Results:\n');
    fprintf('  Noise Level | Physics Error | Vanilla Error | Improvement\n');
    fprintf('  ------------------------------------\n');
    for n = 1:length(noise_levels)
        fprintf('  %.3f       | %.4f        | %.4f       | %.2f%%\n', ...
                noise_levels(n), avg_physics_errors(n), avg_vanilla_errors(n), improvement(n));
    end
    
    % Save results
    noise_results = struct();
    noise_results.noise_levels = noise_levels;
    noise_results.physics_errors = physics_errors;
    noise_results.vanilla_errors = vanilla_errors;
    noise_results.avg_physics_errors = avg_physics_errors;
    noise_results.avg_vanilla_errors = avg_vanilla_errors;
    noise_results.improvement = improvement;
    save('noise_robustness.mat', 'noise_results');
end

function test_parameter_uncertainty(physics_model, vanilla_model, t, params)
    % Generate baseline test trajectories with nominal parameters
    num_test = 20;
    [x0_test, xTest_nominal] = generate_trajectories(num_test, t, params);
    
    % Parameter variation levels (as percentage of nominal values)
    variation_levels = [0.05, 0.1, 0.15, 0.2];  % 5%, 10%, 15%, 20%
    physics_errors = zeros(length(variation_levels), num_test);
    vanilla_errors = zeros(length(variation_levels), num_test);
    
    for v = 1:length(variation_levels)
        var_level = variation_levels(v);
        fprintf('  Testing parameter variation: %.0f%%\n', var_level*100);
        
        for i = 1:num_test
            % Create perturbed parameters
            perturbed_params = perturb_parameters(params, var_level);
            
            % Generate trajectory with perturbed parameters
            [~, x_perturbed] = ode45(@(t,x) FFSM_dynamics(t, x, zeros(2,1), perturbed_params), t, x0_test(:,i));
            x_perturbed = x_perturbed';
            
            % Predict using models trained on nominal parameters
            x_physics = predict_trajectory(t, x0_test(:,i), physics_model);
            x_vanilla = predict_vanilla_trajectory(t, x0_test(:,i), vanilla_model);
            
            % Calculate errors relative to perturbed system (this tests model generalization)
            physics_errors(v,i) = mean(vecnorm(x_physics - x_perturbed, 2, 1) ./ vecnorm(x_perturbed, 2, 1));
            vanilla_errors(v,i) = mean(vecnorm(x_vanilla - x_perturbed, 2, 1) ./ vecnorm(x_perturbed, 2, 1));
        end
    end
    
    % Calculate average errors
    avg_physics_errors = mean(physics_errors, 2);
    avg_vanilla_errors = mean(vanilla_errors, 2);
    
    % Plot results
    figure('Name', 'Parameter Uncertainty Robustness');
    plot(variation_levels*100, avg_physics_errors, 'b-o', ...
         variation_levels*100, avg_vanilla_errors, 'r-o', 'LineWidth', 1.5);
    title('Robustness to Parameter Uncertainty');
    xlabel('Parameter Variation (%)');
    ylabel('Normalized Trajectory Error');
    legend('Physics-Informed', 'Vanilla NN', 'Location', 'northwest');
    grid on;
    saveas(gcf, 'parameter_robustness.png');
    
    % Calculate improvement
    improvement = (avg_vanilla_errors - avg_physics_errors) ./ avg_vanilla_errors * 100;
    
    % Print results
    fprintf('\n  Parameter Uncertainty Robustness Results:\n');
    fprintf('  Variation | Physics Error | Vanilla Error | Improvement\n');
    fprintf('  ------------------------------------\n');
    for v = 1:length(variation_levels)
        fprintf('  %.0f%%      | %.4f        | %.4f       | %.2f%%\n', ...
                variation_levels(v)*100, avg_physics_errors(v), avg_vanilla_errors(v), improvement(v));
    end
    
    % Save results
    param_results = struct();
    param_results.variation_levels = variation_levels;
    param_results.physics_errors = physics_errors;
    param_results.vanilla_errors = vanilla_errors;
    param_results.avg_physics_errors = avg_physics_errors;
    param_results.avg_vanilla_errors = avg_vanilla_errors;
    param_results.improvement = improvement;
    save('parameter_robustness.mat', 'param_results');
end

function perturbed_params = perturb_parameters(params, variation_level)
    % Create a copy of the parameters
    perturbed_params = params;
    
    % List of field names to perturb
    fields = {'m0', 'm1', 'm2', 'I0', 'I1', 'I2', 'l1', 'l2'};
    
    % Perturb each parameter
    for i = 1:length(fields)
        % Random perturbation within the specified variation level
        perturbation = 1 + (2*rand - 1) * variation_level;
        perturbed_params.(fields{i}) = params.(fields{i}) * perturbation;
    end
    
    % Recalculate dependent parameters
    perturbed_params.lc1 = perturbed_params.l1/2;
    perturbed_params.lc2 = perturbed_params.l2/2;
end

function test_external_disturbances(physics_model, vanilla_model, t, params)
    % Generate test trajectories
    num_test = 20;
    [x0_test, ~] = generate_trajectories(num_test, t, params);
    
    % Disturbance amplitudes to test
    disturbance_levels = [0.1, 0.5, 1.0, 2.0];
    physics_errors = zeros(length(disturbance_levels), num_test);
    vanilla_errors = zeros(length(disturbance_levels), num_test);
    
    for d = 1:length(disturbance_levels)
        dist_level = disturbance_levels(d);
        fprintf('  Testing disturbance level: %.1f N\n', dist_level);
        
        for i = 1:num_test
            % Generate disturbed trajectory with external forces
            [~, x_disturbed] = ode45(@(t,x) FFSM_dynamics_with_disturbance(t, x, zeros(2,1), params, dist_level), t, x0_test(:,i));
            x_disturbed = x_disturbed';
            
            % Predict using models (without disturbance knowledge)
            x_physics = predict_trajectory(t, x0_test(:,i), physics_model);
            x_vanilla = predict_vanilla_trajectory(t, x0_test(:,i), vanilla_model);
            
            % Calculate errors
            physics_errors(d,i) = mean(vecnorm(x_physics - x_disturbed, 2, 1) ./ vecnorm(x_disturbed, 2, 1));
            vanilla_errors(d,i) = mean(vecnorm(x_vanilla - x_disturbed, 2, 1) ./ vecnorm(x_disturbed, 2, 1));
        end
    end
    
    % Calculate average errors
    avg_physics_errors = mean(physics_errors, 2);
    avg_vanilla_errors = mean(vanilla_errors, 2);
    
    % Plot results
    figure('Name', 'External Disturbance Robustness');
    plot(disturbance_levels, avg_physics_errors, 'b-o', ...
         disturbance_levels, avg_vanilla_errors, 'r-o', 'LineWidth', 1.5);
    title('Robustness to External Disturbances');
    xlabel('Disturbance Amplitude (N)');
    ylabel('Normalized Trajectory Error');
    legend('Physics-Informed', 'Vanilla NN', 'Location', 'northwest');
    grid on;
    saveas(gcf, 'disturbance_robustness.png');
    
    % Calculate improvement
    improvement = (avg_vanilla_errors - avg_physics_errors) ./ avg_vanilla_errors * 100;
    
    % Print results
    fprintf('\n  External Disturbance Robustness Results:\n');
    fprintf('  Disturbance | Physics Error | Vanilla Error | Improvement\n');
    fprintf('  ------------------------------------\n');
    for d = 1:length(disturbance_levels)
        fprintf('  %.1f N      | %.4f        | %.4f       | %.2f%%\n', ...
                disturbance_levels(d), avg_physics_errors(d), avg_vanilla_errors(d), improvement(d));
    end
    
    % Save results
    disturbance_results = struct();
    disturbance_results.disturbance_levels = disturbance_levels;
    disturbance_results.physics_errors = physics_errors;
    disturbance_results.vanilla_errors = vanilla_errors;
    disturbance_results.avg_physics_errors = avg_physics_errors;
    disturbance_results.avg_vanilla_errors = avg_vanilla_errors;
    disturbance_results.improvement = improvement;
    save('disturbance_robustness.mat', 'disturbance_results');
end

function dx = FFSM_dynamics_with_disturbance(t, x, u, params, dist_level)
    % Regular dynamics
    dx = FFSM_dynamics(t, x, u, params);
    
    % Add random disturbance to the base
    if t > 2 && t < 8  % Apply disturbance in the middle of the trajectory
        % Random disturbance direction
        dist_angle = 2*pi*rand;
        
        % Add disturbance to base acceleration
        dx(6) = dx(6) + dist_level * cos(dist_angle) / params.m0;  % x acceleration
        dx(7) = dx(7) + dist_level * sin(dist_angle) / params.m0;  % y acceleration
        
        % Add random torque disturbance
        dx(8) = dx(8) + (2*rand-1) * dist_level * 0.1 / params.I0;  % angular acceleration
    end
end

function test_extreme_initial_conditions(physics_model, vanilla_model, t, params)
    % Define extreme initial condition ranges
    % Normal ranges were defined in the original code as:
    % x0(1:3) = (rand(3,1) - 0.5);           % Position and orientation
    % x0(4:5) = (rand(2,1) - 0.5) * pi/2;    % Joint angles
    % x0(6:7) = (rand(2,1) - 0.5) * 0.6;     % Linear velocities
    % x0(8:10) = (rand(3,1) - 0.5) * 0.2;    % Angular velocities
    
    % We'll define extreme ranges as multiples of the normal ranges
    extreme_factors = [1.0, 1.5, 2.0, 3.0];  % 1.0 is baseline, 3.0 is 3x more extreme
    
    num_test = 20;
    physics_errors = zeros(length(extreme_factors), num_test);
    vanilla_errors = zeros(length(extreme_factors), num_test);
    
    for e = 1:length(extreme_factors)
        factor = extreme_factors(e);
        fprintf('  Testing initial condition extremity factor: %.1f\n', factor);
        
        for i = 1:num_test
            % Generate extreme initial condition
            x0 = zeros(10, 1);
            x0(1:3) = (rand(3,1) - 0.5) * factor;                 % Position and orientation
            x0(4:5) = (rand(2,1) - 0.5) * pi/2 * factor;          % Joint angles
            x0(6:7) = (rand(2,1) - 0.5) * 0.6 * factor;           % Linear velocities
            x0(8:10) = (rand(3,1) - 0.5) * 0.2 * factor;          % Angular velocities
            
            % Generate true trajectory with extreme initial condition
            [~, x_extreme] = ode45(@(t,x) FFSM_dynamics(t, x, zeros(2,1), params), t, x0);
            x_extreme = x_extreme';
            
            % Predict trajectories
            x_physics = predict_trajectory(t, x0, physics_model);
            x_vanilla = predict_vanilla_trajectory(t, x0, vanilla_model);
            
            % Calculate errors
            physics_errors(e,i) = mean(vecnorm(x_physics - x_extreme, 2, 1) ./ vecnorm(x_extreme, 2, 1));
            vanilla_errors(e,i) = mean(vecnorm(x_vanilla - x_extreme, 2, 1) ./ vecnorm(x_extreme, 2, 1));
        end
    end
    
    % Calculate average errors
    avg_physics_errors = mean(physics_errors, 2);
    avg_vanilla_errors = mean(vanilla_errors, 2);
    
    % Plot results
    figure('Name', 'Extreme Initial Conditions Robustness');
    plot(extreme_factors, avg_physics_errors, 'b-o', ...
         extreme_factors, avg_vanilla_errors, 'r-o', 'LineWidth', 1.5);
    title('Robustness to Extreme Initial Conditions');
    xlabel('Extremity Factor');
    ylabel('Normalized Trajectory Error');
    legend('Physics-Informed', 'Vanilla NN', 'Location', 'northwest');
    grid on;
    saveas(gcf, 'extreme_ic_robustness.png');
    
    % Calculate improvement
    improvement = (avg_vanilla_errors - avg_physics_errors) ./ avg_vanilla_errors * 100;
    
    % Print results
    fprintf('\n  Extreme Initial Conditions Robustness Results:\n');
    fprintf('  Factor | Physics Error | Vanilla Error | Improvement\n');
    fprintf('  ------------------------------------\n');
    for e = 1:length(extreme_factors)
        fprintf('  %.1f    | %.4f        | %.4f       | %.2f%%\n', ...
                extreme_factors(e), avg_physics_errors(e), avg_vanilla_errors(e), improvement(e));
    end
    
    % Save results
    extreme_ic_results = struct();
    extreme_ic_results.extreme_factors = extreme_factors;
    extreme_ic_results.physics_errors = physics_errors;
    extreme_ic_results.vanilla_errors = vanilla_errors;
    extreme_ic_results.avg_physics_errors = avg_physics_errors;
    extreme_ic_results.avg_vanilla_errors = avg_vanilla_errors;
    extreme_ic_results.improvement = improvement;
    save('extreme_ic_robustness.mat', 'extreme_ic_results');
    
    % Also generate a more detailed visualization for the most extreme case
    visualize_extreme_case(physics_model, vanilla_model, t, params, extreme_factors(end));
end

function visualize_extreme_case(physics_model, vanilla_model, t, params, factor)
    % Generate a single extreme initial condition for visualization
    x0 = zeros(10, 1);
    x0(1:3) = (rand(3,1) - 0.5) * factor;                 % Position and orientation
    x0(4:5) = (rand(2,1) - 0.5) * pi/2 * factor;          % Joint angles
    x0(6:7) = (rand(2,1) - 0.5) * 0.6 * factor;           % Linear velocities
    x0(8:10) = (rand(3,1) - 0.5) * 0.2 * factor;          % Angular velocities
    
    % Generate true trajectory
    [~, x_extreme] = ode45(@(t,x) FFSM_dynamics(t, x, zeros(2,1), params), t, x0);
    x_extreme = x_extreme';
    
    % Predict trajectories
    x_physics = predict_trajectory(t, x0, physics_model);
    x_vanilla = predict_vanilla_trajectory(t, x0, vanilla_model);
    
    % Create figure
    figure('Name', 'Extreme Initial Condition Example', 'Position', [100 100 1200 800]);
    
    % Plot base trajectory
    subplot(3, 2, 1);
    plot(x_extreme(1,:), x_extreme(2,:), 'k--', ...
         x_physics(1,:), x_physics(2,:), 'b-', ...
         x_vanilla(1,:), x_vanilla(2,:), 'r-', 'LineWidth', 1.5);
    title('Base Position - Extreme Case');
    xlabel('x (m)'); ylabel('y (m)');
    legend('Ground Truth', 'Physics-Informed', 'Vanilla NN', 'Location', 'best');
    grid on;
    
    % Plot orientation
    subplot(3, 2, 2);
    plot(t, x_extreme(3,:), 'k--', ...
         t, x_physics(3,:), 'b-', ...
         t, x_vanilla(3,:), 'r-', 'LineWidth', 1.5);
    title('Base Orientation - Extreme Case');
    xlabel('Time (s)'); ylabel('Orientation (rad)');
    grid on;
    
    % Plot joint angles
    subplot(3, 2, 3);
    plot(t, x_extreme(4,:), 'k--', ...
         t, x_physics(4,:), 'b-', ...
         t, x_vanilla(4,:), 'r-', 'LineWidth', 1.5);
    title('Joint 1 Angle - Extreme Case');
    xlabel('Time (s)'); ylabel('Angle (rad)');
    grid on;
    
    subplot(3, 2, 4);
    plot(t, x_extreme(5,:), 'k--', ...
         t, x_physics(5,:), 'b-', ...
         t, x_vanilla(5,:), 'r-', 'LineWidth', 1.5);
    title('Joint 2 Angle - Extreme Case');
    xlabel('Time (s)'); ylabel('Angle (rad)');
    grid on;
    
    % Plot linear velocities
    subplot(3, 2, 5);
    plot(t, vecnorm(x_extreme(6:7,:), 2, 1), 'k--', ...
         t, vecnorm(x_physics(6:7,:), 2, 1), 'b-', ...
         t, vecnorm(x_vanilla(6:7,:), 2, 1), 'r-', 'LineWidth', 1.5);
    title('Base Linear Velocity Magnitude - Extreme Case');
    xlabel('Time (s)'); ylabel('Velocity (m/s)');
    grid on;
    
    % Plot angular velocities
    subplot(3, 2, 6);
    plot(t, vecnorm(x_extreme(8:10,:), 2, 1), 'k--', ...
         t, vecnorm(x_physics(8:10,:), 2, 1), 'b-', ...
         t, vecnorm(x_vanilla(8:10,:), 2, 1), 'r-', 'LineWidth', 1.5);
    title('Angular Velocity Magnitude - Extreme Case');
    xlabel('Time (s)'); ylabel('Angular Velocity (rad/s)');
    grid on;
    
    % Save figure
    saveas(gcf, 'extreme_case_visualization.png');
end

function x_pred = predict_vanilla_trajectory(t, x0, net_params)
    n_steps = length(t);
    x_pred = zeros(10, n_steps);
    x_pred(:,1) = x0;
    
    dt = t(2) - t(1);
    for i = 1:n_steps-1
        tspan = [0, dt];
        current_state = dlarray(x_pred(:,i));
        next_state = dlode45(@vanilla_ode_function, tspan, current_state, net_params, 'DataFormat', "CB");
        x_pred(:,i+1) = extractdata(next_state(:,end));
    end
end

function dx = vanilla_ode_function(t, x, params)
    % Neural network layers - same as the physics-informed model
    h1 = tanh(params.fc1.Weights * x + params.fc1.Bias);
    h2 = tanh(params.fc2.Weights * h1 + params.fc2.Bias);
    h3 = tanh(params.fc3.Weights * h2 + params.fc3.Bias);
    dx = params.fc4.Weights * h3 + params.fc4.Bias;
    
    % No physics constraints are applied here
    % This is the key difference from the physics-informed approach
end





function [x0_set, trajectories] = generate_trajectories(num_trajs, t, params)
    x0_set = zeros(10, num_trajs);
    trajectories = cell(num_trajs, 1);
    
    for i = 1:num_trajs
        % Generate random initial state
        x0 = zeros(10, 1);
        x0(1:3) = (rand(3,1) - 0.5);           % Position and orientation
        x0(4:5) = (rand(2,1) - 0.5) * pi/2;    % Joint angles
        x0(6:7) = (rand(2,1) - 0.5) * 0.6;     % Linear velocities
        x0(8:10) = (rand(3,1) - 0.5) * 0.2;    % Angular velocities
        
        x0_set(:,i) = x0;
        
        % Simulate trajectory
        [~, x] = ode45(@(t,x) FFSM_dynamics(t, x, zeros(2,1), params), t, x0);
        trajectories{i} = x';
    end
end




function predictions = neural_ode_forward(t, X0, net_params)
    tspan = [t(1), t(end)];
    predictions = dlode45(@ode_function, tspan, X0, net_params, 'DataFormat', "CB");
end

function dx = ode_function(t, x, params)
    % Neural network layers
    h1 = tanh(params.fc1.Weights * x + params.fc1.Bias);
    h2 = tanh(params.fc2.Weights * h1 + params.fc2.Bias);
    h3 = tanh(params.fc3.Weights * h2 + params.fc3.Bias);
    dx = params.fc4.Weights * h3 + params.fc4.Bias;
    
    % Physics-informed constraints
    dx(1:5) = x(6:10);  % Position derivatives are velocities
    
    % Scale acceleration terms
    dx(6:7) = dx(6:7) * 0.01;     % Base linear accelerations
    dx(8) = dx(8) * 0.005;        % Base angular acceleration
    dx(9:10) = dx(9:10) * 0.005;  % Joint accelerations
end



function dx = FFSM_dynamics(t, x, u, params)
    % Extract states
    xb = x(1); yb = x(2);
    theta0 = x(3);
    q1 = x(4); q2 = x(5);
    dxb = x(6); dyb = x(7);
    dtheta0 = x(8);
    dq1 = x(9); dq2 = x(10);
    
    % Mass Matrix Elements
    M11 = params.m0 + params.m1 + params.m2;
    M12 = 0;
    M13 = -params.m1*(params.lc1*sin(theta0 + q1)) - params.m2*(params.l1*sin(theta0 + q1) + params.lc2*sin(theta0 + q1 + q2));
    M14 = -params.m1*params.lc1*sin(theta0 + q1) - params.m2*(params.l1*sin(theta0 + q1) + params.lc2*sin(theta0 + q1 + q2));
    M15 = -params.m2*params.lc2*sin(theta0 + q1 + q2);
    M22 = params.m0 + params.m1 + params.m2;
    M23 = params.m1*(params.lc1*cos(theta0 + q1)) + params.m2*(params.l1*cos(theta0 + q1) + params.lc2*cos(theta0 + q1 + q2));
    M24 = params.m1*params.lc1*cos(theta0 + q1) + params.m2*(params.l1*cos(theta0 + q1) + params.lc2*cos(theta0 + q1 + q2));
    M25 = params.m2*params.lc2*cos(theta0 + q1 + q2);
    
    M33 = params.I0 + params.I1 + params.I2 + params.m1*params.lc1^2 + params.m2*(params.l1^2 + params.lc2^2 + 2*params.l1*params.lc2*cos(q2));
    M34 = params.I1 + params.I2 + params.m1*params.lc1^2 + params.m2*(params.l1^2 + params.lc2^2 + 2*params.l1*params.lc2*cos(q2));
    M35 = params.I2 + params.m2*params.lc2^2 + params.m2*params.l1*params.lc2*cos(q2);
    M44 = params.I1 + params.I2 + params.m1*params.lc1^2 + params.m2*(params.l1^2 + params.lc2^2 + 2*params.l1*params.lc2*cos(q2));
    M45 = params.I2 + params.m2*params.lc2^2 + params.m2*params.l1*params.lc2*cos(q2);
    M55 = params.I2 + params.m2*params.lc2^2;
    
    % Form complete mass matrix
    M = [M11 M12 M13 M14 M15;
         M12 M22 M23 M24 M25;
         M13 M23 M33 M34 M35;
         M14 M24 M34 M44 M45;
         M15 M25 M35 M45 M55];
    
    % Coriolis terms
    h = params.m2*params.l1*params.lc2*sin(q2);
    C = zeros(5,5);
    
    C13 = -h*(dq1 + dq2);
    C14 = -h*dq2;
    C15 = -h*dq2;
    C23 = -h*(dq1 + dq2);
    C24 = -h*dq2;
    C25 = -h*dq2;
    C33 = -h*dq2;
    C34 = -h*dq2;
    C35 = -h*(dq1 + dq2);
    C43 = -h*dq2;
    C44 = -h*dq2;
    C45 = -h*(dq1 + dq2);
    C53 = h*dq1;
    C54 = h*dq1;
    
    C = [0 0 C13 C14 C15;
         0 0 C23 C24 C25;
         0 0 C33 C34 C35;
         0 0 C43 C44 C45;
         0 0 C53 C54 0];
    
    % Input mapping matrix
    B = [0 0;
         0 0;
         0 0;
         1 0;
         0 1];
    
    % State vectors
    q = [xb; yb; theta0; q1; q2];
    dq = [dxb; dyb; dtheta0; dq1; dq2];
    
    % System dynamics
    ddq = M\(B*u - C*dq);
    
    % State derivatives
    dx = [dq; ddq];
end






function x_pred = predict_trajectory(t, x0, net_params)
    n_steps = length(t);
    x_pred = zeros(10, n_steps);
    x_pred(:,1) = x0;
    
    dt = t(2) - t(1);
    for i = 1:n_steps-1
        tspan = [0, dt];
        current_state = dlarray(x_pred(:,i));
        next_state = dlode45(@ode_function, tspan, current_state, net_params, 'DataFormat', "CB");
        x_pred(:,i+1) = extractdata(next_state(:,end));
    end
end
