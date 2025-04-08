function main_free_floating_neural_ode()
    % System parameters - Only base, no links
    params = struct();
    params.m0 = 19;       % Mass of the base
    params.I0 = 0.456;    % Inertia of the base

    % Simulation parameters
    T = 10;
    dt = 0.01;
    t = 0:dt:T;
    
    % Generate training data
    num_train = 10;
    fprintf('Generating %d training trajectories...\n', num_train);
    [x0_train, xTrain] = generate_trajectories(num_train, t, params);

    % Neural ODE setup
    net_params = initialize_network(6, 256);  % Reduced state size to 6

    % Training parameters
    train_params = struct();
    train_params.num_iter = 1000;
    train_params.batch_size = 32;
    train_params.seq_length = 50;
    train_params.learning_rate = 0.0005;
    train_params.plot_freq = 100;

    % Train the model
    fprintf('Starting training...\n');
    [net_params, losses] = train_model(net_params, xTrain, t, train_params);
    
    % Plot final training progress
    plot_training_progress(losses);

    % Generate and evaluate test trajectories
    num_test = 10;
    fprintf('Testing on %d new trajectories...\n', num_test);
    [x0_test, xTest] = generate_trajectories(num_test, t, params);
    evaluate_model(net_params, xTest, x0_test, t);

    % Save trained model and losses
    save('physics_informed_model.mat', 'net_params');
    save('physics_informed_losses.mat', 'losses');
end

function [x0_set, trajectories] = generate_trajectories(num_trajs, t, params)
    x0_set = zeros(6, num_trajs);  % Reduced to 6 states
    trajectories = cell(num_trajs, 1);
    
    for i = 1:num_trajs
        % Generate random initial state:
        % [x, y, theta, dx, dy, dtheta]
        x0 = zeros(6, 1);
        x0(1:3) = (rand(3,1) - 0.5);           % Position and orientation
        x0(4:6) = (rand(3,1) - 0.5) * 0.6;     % Linear and angular velocities
        
        x0_set(:,i) = x0;
        
        % Simulate trajectory - no control inputs for free-floating base
        [~, x] = ode45(@(t,x) base_only_dynamics(t, x, params), t, x0);
        trajectories{i} = x';
    end
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

function [net_params, losses] = train_model(net_params, xTrain, t, train_params)
    % Initialize storage for losses
    losses = zeros(train_params.num_iter, 1);
    
    % Initialize Adam optimizer parameters
    averageGrad = [];
    averageSqGrad = [];
    gradDecay = 0.9;
    sqGradDecay = 0.999;
    
    % Create figure for training visualization
    figure('Name', 'Training Progress', 'Position', [100 100 800 600]);
    
    % Training loop
    for iteration = 1:train_params.num_iter
        % Create batch
        [X0, targets] = create_batch(xTrain, train_params.batch_size, train_params.seq_length);
        
        % Compute loss and gradients
        [loss, gradients] = dlfeval(@compute_loss, t(1:train_params.seq_length), X0, net_params, targets);
        
        % Update parameters
        [net_params, averageGrad, averageSqGrad] = adamupdate(net_params, gradients, ...
            averageGrad, averageSqGrad, iteration, train_params.learning_rate, gradDecay, sqGradDecay);
        
        % Store loss
        losses(iteration) = double(gather(extractdata(loss)));
        
        % Display progress
        if mod(iteration, train_params.plot_freq) == 0
            fprintf('Iteration: %d, Loss: %.4f\n', iteration, losses(iteration));
            
            % Update training progress plot
            subplot(2,1,1);
            plot(1:iteration, losses(1:iteration), 'b-', 'LineWidth', 1.5);
            title('Training Loss');
            xlabel('Iteration');
            ylabel('Loss');
            grid on;
            
            drawnow;
        end
    end
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

function [loss, gradients] = compute_loss(t, X0, net_params, targets)
    % Forward pass
    predictions = neural_ode_forward(t, X0, net_params);
    
    % Split predictions for different loss components
    pos_pred = predictions(1:3,:,:);    % Position and orientation
    vel_pred = predictions(4:6,:,:);    % Velocities
    pos_true = targets(1:3,:,:);
    vel_true = targets(4:6,:,:);
    
    % Position and orientation loss
    pos_loss = mean(abs(pos_pred - pos_true), 'all');
    
    % Velocity loss
    vel_loss = mean(abs(vel_pred - vel_true), 'all');
    
    % Momentum conservation loss
    momentum_loss = compute_momentum_loss(predictions);
    
    % Combined loss with weights
    loss = pos_loss + 2*vel_loss + 5*momentum_loss;
    
    % Compute gradients
    gradients = dlgradient(loss, net_params);
end

function momentum_loss = compute_momentum_loss(predictions)
    % Get initial velocities (linear and angular)
    linear_vel_init = predictions(4:5,:,1);
    angular_vel_init = predictions(6,:,1);
    
    % Compute deviation from initial values (conservation of momentum)
    linear_vel_loss = mean(abs(predictions(4:5,:,:) - linear_vel_init), 'all');
    angular_vel_loss = mean(abs(predictions(6,:,:) - angular_vel_init), 'all');
    
    momentum_loss = linear_vel_loss + angular_vel_loss;
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
    dx(1:3) = x(4:6);  % Position derivatives are velocities
    
    % Scale acceleration terms
    dx(4:5) = dx(4:5) * 0.01;     % Base linear accelerations
    dx(6) = dx(6) * 0.005;        % Base angular acceleration
end

function evaluate_model(net_params, xTest, x0_test, t)
    num_test = size(x0_test, 2);
    errors = zeros(6, num_test);
    
    % Plot base motion for first 4 test cases
    figure('Name', 'Base Motion Test Cases', 'Position', [100 100 800 600]);
    for i = 1:min(4, num_test)
        subplot(2,2,i)
        x_true = xTest{i};
        x_pred = predict_trajectory(t, x0_test(:,i), net_params);
        
        plot(x_true(1,:), x_true(2,:), 'r--', x_pred(1,:), x_pred(2,:), 'b-', 'LineWidth', 1.5)
        title(sprintf('Test Case %d - Base Motion', i))
        xlabel('x (m)'); ylabel('y (m)')
        legend('True', 'Predicted', 'Location', 'best')
        grid on
        
        errors(:,i) = mean(abs(x_true - x_pred), 2);
    end
    
    % Plot all states for first test case
    figure('Name', 'All States - Test Case 1', 'Position', [100 100 1200 800]);
    x_true = xTest{1};
    x_pred = predict_trajectory(t, x0_test(:,1), net_params);
    
    % Base positions and orientation
    subplot(2,3,1)
    plot(t, x_true(1,:), 'r--', t, x_pred(1,:), 'b-', 'LineWidth', 1.5)
    title('Base Position x')
    xlabel('Time (s)'); ylabel('x (m)')
    grid on
    
    subplot(2,3,2)
    plot(t, x_true(2,:), 'r--', t, x_pred(2,:), 'b-', 'LineWidth', 1.5)
    title('Base Position y')
    xlabel('Time (s)'); ylabel('y (m)')
    grid on
    
    subplot(2,3,3)
    plot(t, x_true(3,:), 'r--', t, x_pred(3,:), 'b-', 'LineWidth', 1.5)
    title('Base Orientation')
    xlabel('Time (s)'); ylabel('\theta_0 (rad)')
    grid on
    
    % Velocities
    subplot(2,3,4)
    plot(t, x_true(4,:), 'r--', t, x_pred(4,:), 'b-', 'LineWidth', 1.5)
    title('Base Velocity dx')
    xlabel('Time (s)'); ylabel('dx (m/s)')
    grid on
    
    subplot(2,3,5)
    plot(t, x_true(5,:), 'r--', t, x_pred(5,:), 'b-', 'LineWidth', 1.5)
    title('Base Velocity dy')
    xlabel('Time (s)'); ylabel('dy (m/s)')
    grid on
    
    subplot(2,3,6)
    plot(t, x_true(6,:), 'r--', t, x_pred(6,:), 'b-', 'LineWidth', 1.5)
    title('Base Angular Velocity')
    xlabel('Time (s)'); ylabel('d\theta_0 (rad/s)')
    grid on
    
    % Display errors
    fprintf('\nTest Results:\n');
    for i = 1:num_test
        fprintf('\nTest Case %d:\n', i);
        fprintf('Position (x,y): %.4f, %.4f\n', errors(1:2,i));
        fprintf('Orientation: %.4f\n', errors(3,i));
        fprintf('Base Velocities: %.4f, %.4f\n', errors(4:5,i));
        fprintf('Angular Velocity: %.4f\n', errors(6,i));
    end
end

function plot_training_progress(losses)
    figure('Name', 'Final Training Progress');
    plot(losses, 'b-', 'LineWidth', 1.5);
    title('Training Loss History');
    xlabel('Iteration');
    ylabel('Loss');
    grid on;
end

function x_pred = predict_trajectory(t, x0, net_params)
    n_steps = length(t);
    x_pred = zeros(6, n_steps);
    x_pred(:,1) = x0;
    
    dt = t(2) - t(1);
    for i = 1:n_steps-1
        tspan = [0, dt];
        current_state = dlarray(x_pred(:,i));
        next_state = dlode45(@ode_function, tspan, current_state, net_params, 'DataFormat', "CB");
        x_pred(:,i+1) = extractdata(next_state(:,end));
    end
end

function dx = base_only_dynamics(t, x, params)
    % State vector components:
    % x = [x_pos; y_pos; theta; dx; dy; dtheta]
    
    % For a free-floating base in space with no external forces,
    % linear and angular momentum are conserved.
    % This means accelerations (ddx, ddy, ddtheta) are all zero.
    
    % State derivatives (velocity is constant, position changes with velocity)
    dx = zeros(6, 1);
    dx(1) = x(4);  % dx = velocity_x
    dx(2) = x(5);  % dy = velocity_y
    dx(3) = x(6);  % dtheta = angular_velocity
    dx(4) = 0;     % ddx = 0 (no acceleration)
    dx(5) = 0;     % ddy = 0 (no acceleration)
    dx(6) = 0;     % ddtheta = 0 (no angular acceleration)
end
