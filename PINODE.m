function main_free_floating_neural_ode()
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
    
    % Generate training data
    num_train = 100;
    fprintf('Generating %d training trajectories...\n', num_train);
    [x0_train, xTrain] = generate_trajectories(num_train, t, params);

    % Neural ODE setup
    net_params = initialize_network(10, 256);

    % Training parameters
    train_params = struct();
    train_params.num_iter = 5000;
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

    % At the end of main_free_floating_neural_ode function, after training is complete
    save('physics_informed_model.mat', 'net_params');
    save('physics_informed_losses.mat', 'losses');

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
    pos_pred = predictions(1:5,:,:);
    vel_pred = predictions(6:10,:,:);
    pos_true = targets(1:5,:,:);
    vel_true = targets(6:10,:,:);
    
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
    % Get initial velocities
    linear_vel_init = predictions(6:7,:,1);
    angular_vel_init = predictions(8,:,1);
    
    % Compute deviation from initial values
    linear_vel_loss = mean(abs(predictions(6:7,:,:) - linear_vel_init), 'all');
    angular_vel_loss = mean(abs(predictions(8,:,:) - angular_vel_init), 'all');
    
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
    dx(1:5) = x(6:10);  % Position derivatives are velocities
    
    % Scale acceleration terms
    dx(6:7) = dx(6:7) * 0.01;     % Base linear accelerations
    dx(8) = dx(8) * 0.005;        % Base angular acceleration
    dx(9:10) = dx(9:10) * 0.005;  % Joint accelerations
end

function evaluate_model(net_params, xTest, x0_test, t)
    num_test = size(x0_test, 2);
    errors = zeros(10, num_test);
    
    % Plot base motion for first 4 test cases
    figure('Name', 'Base Motion Test Cases', 'Position', [100 100 800 600]);
    for i = 1:4
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
    subplot(3,3,1)
    plot(t, x_true(1,:), 'r--', t, x_pred(1,:), 'b-', 'LineWidth', 1.5)
    title('Base Position x')
    xlabel('Time (s)'); ylabel('x (m)')
    grid on
    
    subplot(3,3,2)
    plot(t, x_true(2,:), 'r--', t, x_pred(2,:), 'b-', 'LineWidth', 1.5)
    title('Base Position y')
    xlabel('Time (s)'); ylabel('y (m)')
    grid on
    
    subplot(3,3,3)
    plot(t, x_true(3,:), 'r--', t, x_pred(3,:), 'b-', 'LineWidth', 1.5)
    title('Base Orientation')
    xlabel('Time (s)'); ylabel('\theta_0 (rad)')
    grid on
    
    % Joint angles
    subplot(3,3,4)
    plot(t, x_true(4,:), 'r--', t, x_pred(4,:), 'b-', 'LineWidth', 1.5)
    title('Joint Angle q1')
    xlabel('Time (s)'); ylabel('q1 (rad)')
    grid on
    
    subplot(3,3,5)
    plot(t, x_true(5,:), 'r--', t, x_pred(5,:), 'b-', 'LineWidth', 1.5)
    title('Joint Angle q2')
    xlabel('Time (s)'); ylabel('q2 (rad)')
    grid on
    
    % Velocities
    subplot(3,3,6)
    plot(t, x_true(6,:), 'r--', t, x_pred(6,:), 'b-', 'LineWidth', 1.5)
    title('Base Velocity dx')
    xlabel('Time (s)'); ylabel('dx (m/s)')
    grid on
    
    subplot(3,3,7)
    plot(t, x_true(7,:), 'r--', t, x_pred(7,:), 'b-', 'LineWidth', 1.5)
    title('Base Velocity dy')
    xlabel('Time (s)'); ylabel('dy (m/s)')
    grid on
    
    subplot(3,3,8)
    plot(t, x_true(8,:), 'r--', t, x_pred(8,:), 'b-', 'LineWidth', 1.5)
    title('Base Angular Velocity')
    xlabel('Time (s)'); ylabel('d\theta_0 (rad/s)')
    grid on
    
    subplot(3,3,9)
    plot(t, x_true(9,:), 'r--', t, x_pred(9,:), 'b-', ...
         t, x_true(10,:), 'r:', t, x_pred(10,:), 'b:', 'LineWidth', 1.5)
    title('Joint Velocities')
    xlabel('Time (s)'); ylabel('Angular Velocity (rad/s)')
    legend('True dq1', 'Pred dq1', 'True dq2', 'Pred dq2', 'Location', 'best')
    grid on
    
    % Display errors
    fprintf('\nTest Results:\n');
    for i = 1:num_test
        fprintf('\nTest Case %d:\n', i);
        fprintf('Position (x,y): %.4f, %.4f\n', errors(1:2,i));
        fprintf('Orientation: %.4f\n', errors(3,i));
        fprintf('Joint Angles: %.4f, %.4f\n', errors(4:5,i));
        fprintf('Base Velocities: %.4f, %.4f\n', errors(6:7,i));
        fprintf('Angular Velocity: %.4f\n', errors(8,i));
        fprintf('Joint Velocities: %.4f, %.4f\n', errors(9:10,i));
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
