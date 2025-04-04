function vanilla_neural_ode_comparison()
    % System parameters - same as the physics-informed approach
    params = struct();
    params.m0 = 60; params.m1 = 6; params.m2 = 5;
    params.I0 = 24; params.I1 = 1; params.I2 = 0.8;
    params.l0 = 0.75; params.l1 = 0.75; params.l2 = 0.75;
    params.lc1 = params.l1/2; params.lc2 = params.l2/2;

    % Simulation parameters
    T = 10;
    dt = 0.01;
    t = 0:dt:T;
    
    % Generate training data - reuse from original code
    num_train = 100;
    fprintf('Generating %d training trajectories for baseline comparison...\n', num_train);
    [x0_train, xTrain] = generate_trajectories(num_train, t, params);

    % Neural ODE setup - same architecture, but without physics constraints
    vanilla_net_params = initialize_network(10, 256);

    % Training parameters
    train_params = struct();
    train_params.num_iter = 5000;
    train_params.batch_size = 32;
    train_params.seq_length = 50;
    train_params.learning_rate = 0.0005;
    train_params.plot_freq = 100;

    % Train the vanilla model
    fprintf('Starting training of vanilla model (no physics constraints)...\n');
    [vanilla_net_params, vanilla_losses] = train_vanilla_model(vanilla_net_params, xTrain, t, train_params);
    
    % Plot training progress
    figure('Name', 'Training Progress Comparison');
    
    % Load the physics-informed model results for comparison
    % You'll need to save these from your main function
    physics_losses = load('physics_informed_losses.mat').losses;
    
    semilogy(1:length(physics_losses), physics_losses, 'b-', ...
             1:length(vanilla_losses), vanilla_losses, 'r-', 'LineWidth', 1.5);
    title('Training Loss Comparison');
    xlabel('Iteration');
    ylabel('Loss (log scale)');
    legend('Physics-Informed Neural ODE', 'Vanilla Neural ODE', 'Location', 'best');
    grid on;
    saveas(gcf, 'loss_comparison.png');

    % Generate and evaluate test trajectories
    num_test = 10;
    fprintf('Testing both models on %d new trajectories...\n', num_test);
    [x0_test, xTest] = generate_trajectories(num_test, t, params);
    
    % Load the physics-informed model parameters
    physics_net_params = load('physics_informed_model.mat').net_params;
    
    % Compare performance
    compare_models(physics_net_params, vanilla_net_params, xTest, x0_test, t, params);
end

function [net_params, losses] = train_vanilla_model(net_params, xTrain, t, train_params)
    % Initialize storage for losses
    losses = zeros(train_params.num_iter, 1);
    
    % Initialize Adam optimizer parameters
    averageGrad = [];
    averageSqGrad = [];
    
    % Create figure for training visualization
    figure('Name', 'Vanilla Model Training Progress');
    
    % Training loop
    for iteration = 1:train_params.num_iter
        % Create batch
        [X0, targets] = create_batch(xTrain, train_params.batch_size, train_params.seq_length);
        
        % Compute loss and gradients - using vanilla loss function
        [loss, gradients] = dlfeval(@compute_vanilla_loss, t(1:train_params.seq_length), X0, net_params, targets);
        
        % Update parameters
        [net_params, averageGrad, averageSqGrad] = adamupdate(net_params, gradients, ...
            averageGrad, averageSqGrad, iteration, train_params.learning_rate);
        
        % Store loss
        losses(iteration) = double(gather(extractdata(loss)));
        
        % Display progress
        if mod(iteration, train_params.plot_freq) == 0
            fprintf('Vanilla Model - Iteration: %d, Loss: %.4f\n', iteration, losses(iteration));
            
            % Update training progress plot
            plot(1:iteration, losses(1:iteration), 'r-', 'LineWidth', 1.5);
            title('Vanilla Neural ODE Training Loss');
            xlabel('Iteration');
            ylabel('Loss');
            grid on;
            drawnow;
        end
    end
    
    % Save the model
    save('vanilla_model.mat', 'net_params');
    save('vanilla_losses.mat', 'losses');
end

function [loss, gradients] = compute_vanilla_loss(t, X0, net_params, targets)
    % Forward pass - using the vanilla ODE function
    predictions = vanilla_neural_ode_forward(t, X0, net_params);
    
    % Simple MSE loss without physics-based components
    loss = mean((predictions - targets).^2, 'all');
    
    % Compute gradients
    gradients = dlgradient(loss, net_params);
end

function predictions = vanilla_neural_ode_forward(t, X0, net_params)
    tspan = [t(1), t(end)];
    predictions = dlode45(@vanilla_ode_function, tspan, X0, net_params, 'DataFormat', "CB");
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

function compare_models(physics_params, vanilla_params, xTest, x0_test, t, sys_params)
    num_test = size(x0_test, 2);
    physics_errors = zeros(10, num_test);
    vanilla_errors = zeros(10, num_test);
    
    % Calculate conservation metrics
    physics_conservation = zeros(num_test, 1);
    vanilla_conservation = zeros(num_test, 1);
    
    % Plot comparison for first 4 test cases
    figure('Name', 'Model Comparison', 'Position', [100 100 1200 800]);
    
    for i = 1:num_test
        % Get true trajectory
        x_true = xTest{i};
        
        % Get predicted trajectories
        x_physics = predict_trajectory(t, x0_test(:,i), physics_params);
        x_vanilla = predict_vanilla_trajectory(t, x0_test(:,i), vanilla_params);
        
        % Calculate errors
        physics_errors(:,i) = mean(abs(x_true - x_physics), 2);
        vanilla_errors(:,i) = mean(abs(x_true - x_vanilla), 2);
        
        % Calculate conservation metric (momentum preservation)
        physics_conservation(i) = calculate_momentum_conservation(x_physics, sys_params);
        vanilla_conservation(i) = calculate_momentum_conservation(x_vanilla, sys_params);
        
        % Plot first 4 test cases
        if i <= 4
            % Plot base position
            subplot(4, 2, (i-1)*2 + 1);
            plot(x_true(1,:), x_true(2,:), 'k--', ...
                 x_physics(1,:), x_physics(2,:), 'b-', ...
                 x_vanilla(1,:), x_vanilla(2,:), 'r-', 'LineWidth', 1.5);
            title(sprintf('Test Case %d - Base Position', i));
            xlabel('x (m)'); ylabel('y (m)');
            legend('Ground Truth', 'Physics-Informed', 'Vanilla NN', 'Location', 'best');
            grid on;
            
            % Plot joint angles
            subplot(4, 2, (i-1)*2 + 2);
            plot(t, x_true(4,:), 'k--', t, x_physics(4,:), 'b-', t, x_vanilla(4,:), 'r-', 'LineWidth', 1.5);
            hold on;
            plot(t, x_true(5,:), 'k:', t, x_physics(5,:), 'b:', t, x_vanilla(5,:), 'r:', 'LineWidth', 1.5);
            hold off;
            title(sprintf('Test Case %d - Joint Angles', i));
            xlabel('Time (s)'); ylabel('Angle (rad)');
            legend('q1 Truth', 'q1 Physics', 'q1 Vanilla', 'q2 Truth', 'q2 Physics', 'q2 Vanilla', 'Location', 'best');
            grid on;
        end
    end
    
    % Save figure
    saveas(gcf, 'model_comparison.png');
    
    % Create error comparison table
    figure('Name', 'Error Comparison');
    
    % Calculate average errors across all test cases
    avg_physics_errors = mean(physics_errors, 2);
    avg_vanilla_errors = mean(vanilla_errors, 2);
    
    % State labels
    state_labels = {'x', 'y', '\theta_0', 'q_1', 'q_2', 'v_x', 'v_y', '\omega_0', '\omega_1', '\omega_2'};
    
    % Bar plot of errors
    bar([avg_physics_errors, avg_vanilla_errors]);
    set(gca, 'XTickLabel', state_labels);
    legend('Physics-Informed', 'Vanilla NN');
    title('Average State Prediction Error');
    ylabel('Mean Absolute Error');
    grid on;
    saveas(gcf, 'error_comparison.png');
    
    % Conservation of momentum comparison
    figure('Name', 'Conservation of Momentum');
    boxplot([physics_conservation, vanilla_conservation], 'Labels', {'Physics-Informed', 'Vanilla NN'});
    title('Conservation of Momentum Metric');
    ylabel('Momentum Conservation Error (lower is better)');
    grid on;
    saveas(gcf, 'conservation_comparison.png');
    
    % Print statistical results
    fprintf('\nModel Comparison Results:\n');
    fprintf('                     Physics-Informed     Vanilla NN     Improvement %%\n');
    fprintf('------------------------------------------------------------------------------------------------\n');
    for i = 1:10
        improvement = (avg_vanilla_errors(i) - avg_physics_errors(i)) / avg_vanilla_errors(i) * 100;
        fprintf('%-15s      %10.4f      %10.4f         %10.2f%%\n', ...
                state_labels{i}, avg_physics_errors(i), avg_vanilla_errors(i), improvement);
    end
    
    % Average improvement across all states
    avg_improvement = (mean(avg_vanilla_errors) - mean(avg_physics_errors)) / mean(avg_vanilla_errors) * 100;
    fprintf('------------------------------------------------------------------------------------------------\n');
    fprintf('%-15s      %10.4f      %10.4f         %10.2f%%\n', ...
            'Overall', mean(avg_physics_errors), mean(avg_vanilla_errors), avg_improvement);
    
    % Conservation metrics
    conserv_improvement = (mean(vanilla_conservation) - mean(physics_conservation)) / mean(vanilla_conservation) * 100;
    fprintf('%-15s      %10.4f      %10.4f         %10.2f%%\n', ...
            'Conservation', mean(physics_conservation), mean(vanilla_conservation), conserv_improvement);
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

function conservation_error = calculate_momentum_conservation(trajectory, params)
    % Extract initial linear and angular momenta
    initial_lin_momentum = calculate_linear_momentum(trajectory(:,1), params);
    initial_ang_momentum = calculate_angular_momentum(trajectory(:,1), params);
    
    % Calculate momenta at each timestep
    n_steps = size(trajectory, 2);
    lin_momentum_error = 0;
    ang_momentum_error = 0;
    
    for i = 1:n_steps
        current_lin_momentum = calculate_linear_momentum(trajectory(:,i), params);
        current_ang_momentum = calculate_angular_momentum(trajectory(:,i), params);
        
        lin_momentum_error = lin_momentum_error + norm(current_lin_momentum - initial_lin_momentum);
        ang_momentum_error = ang_momentum_error + abs(current_ang_momentum - initial_ang_momentum);
    end
    
    % Average error over the trajectory
    conservation_error = (lin_momentum_error + ang_momentum_error) / n_steps;
end

function lin_momentum = calculate_linear_momentum(state, params)
    % Extract state variables
    vx = state(6); vy = state(7);
    theta0 = state(3); q1 = state(4); q2 = state(5);
    omega0 = state(8); omega1 = state(9); omega2 = state(10);
    
    % Calculate total linear momentum
    p_base = [params.m0 * vx; params.m0 * vy];
    
    % Link 1 velocity
    v1_x = vx - params.lc1 * (omega0 + omega1) * sin(theta0 + q1);
    v1_y = vy + params.lc1 * (omega0 + omega1) * cos(theta0 + q1);
    p_link1 = [params.m1 * v1_x; params.m1 * v1_y];
    
    % Link 2 velocity
    v2_x = vx - params.l1 * (omega0 + omega1) * sin(theta0 + q1) - params.lc2 * (omega0 + omega1 + omega2) * sin(theta0 + q1 + q2);
    v2_y = vy + params.l1 * (omega0 + omega1) * cos(theta0 + q1) + params.lc2 * (omega0 + omega1 + omega2) * cos(theta0 + q1 + q2);
    p_link2 = [params.m2 * v2_x; params.m2 * v2_y];
    
    % Total linear momentum
    lin_momentum = norm(p_base + p_link1 + p_link2);
end

function ang_momentum = calculate_angular_momentum(state, params)
    % Extract state variables
    x = state(1); y = state(2);
    vx = state(6); vy = state(7);
    theta0 = state(3); q1 = state(4); q2 = state(5);
    omega0 = state(8); omega1 = state(9); omega2 = state(10);
    
    % Base angular momentum
    h_base = params.I0 * omega0;
    
    % Link 1 angular momentum
    v1_x = vx - params.lc1 * (omega0 + omega1) * sin(theta0 + q1);
    v1_y = vy + params.lc1 * (omega0 + omega1) * cos(theta0 + q1);
    r1_x = x + params.lc1 * cos(theta0 + q1);
    r1_y = y + params.lc1 * sin(theta0 + q1);
    h_link1 = params.I1 * (omega0 + omega1) + params.m1 * (r1_x * v1_y - r1_y * v1_x);
    
    % Link 2 angular momentum
    v2_x = vx - params.l1 * (omega0 + omega1) * sin(theta0 + q1) - params.lc2 * (omega0 + omega1 + omega2) * sin(theta0 + q1 + q2);
    v2_y = vy + params.l1 * (omega0 + omega1) * cos(theta0 + q1) + params.lc2 * (omega0 + omega1 + omega2) * cos(theta0 + q1 + q2);
    r2_x = x + params.l1 * cos(theta0 + q1) + params.lc2 * cos(theta0 + q1 + q2);
    r2_y = y + params.l1 * sin(theta0 + q1) + params.lc2 * sin(theta0 + q1 + q2);
    h_link2 = params.I2 * (omega0 + omega1 + omega2) + params.m2 * (r2_x * v2_y - r2_y * v2_x);
    
    % Total angular momentum
    ang_momentum = abs(h_base + h_link1 + h_link2);
end

% The following functions should be identical to the ones in the main code
% I'm not redefining them to avoid duplication, but in practice you would include:
% - predict_trajectory
% - FFSM_dynamics
% - create_batch 
% - initialize_network
% - generate_trajectories






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




function [X0, targets] = create_batch(xTrain, batch_size, seq_length)
    num_trajectories = length(xTrain);
    state_size = size(xTrain{1}, 1);
    
    X0 = zeros(state_size, batch_size);
    targets = zeros(state_size, batch_size, seq_length);
    
    for i = 1:batch_size
        traj_idx = randi(num_trajectories);
        trajectory = xTrain{traj_idx};
        
        max_start = size(trajectory, 2) - seq_length;
        start_idx = randi(max_start);
        
        X0(:,i) = trajectory(:, start_idx);
        targets(:,i,:) = trajectory(:, start_idx+1:start_idx+seq_length);
    end
    
    X0 = dlarray(X0);
    targets = dlarray(targets);
end




function net_params = initialize_network(state_size, hidden_size)
    net_params = struct;
    
    % Initialize layers with Glorot initialization
    scale1 = sqrt(2/(state_size + hidden_size));
    scale2 = sqrt(2/(hidden_size + hidden_size));
    scale3 = sqrt(2/(hidden_size + state_size));
    
    net_params.fc1 = struct;
    net_params.fc1.Weights = dlarray(randn([hidden_size state_size]) * scale1);
    net_params.fc1.Bias = dlarray(zeros([hidden_size 1]));
    
    net_params.fc2 = struct;
    net_params.fc2.Weights = dlarray(randn([hidden_size hidden_size]) * scale2);
    net_params.fc2.Bias = dlarray(zeros([hidden_size 1]));
    
    net_params.fc3 = struct;
    net_params.fc3.Weights = dlarray(randn([hidden_size hidden_size]) * scale2);
    net_params.fc3.Bias = dlarray(zeros([hidden_size 1]));
    
    net_params.fc4 = struct;
    net_params.fc4.Weights = dlarray(randn([state_size hidden_size]) * scale3);
    net_params.fc4.Bias = dlarray(zeros([state_size 1]));
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
