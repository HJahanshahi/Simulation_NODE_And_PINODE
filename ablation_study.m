function ablation_study()
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
    
    % Load optimal hyperparameters
    % In practice, these would come from hyperparameter_tuning_results.mat
    % For the ablation study, we'll set them directly
    hidden_size = 256;  % Example optimal value
    learning_rate = 0.0005;  % Example optimal value
    momentum_weight = 5;  % Example optimal value
    
    % Generate training and test data
    num_train = 80;
    num_test = 20;
    fprintf('Generating %d training and %d test trajectories for ablation study...\n', num_train, num_test);
    [x0_train, xTrain] = generate_trajectories(num_train, t, params);
    [x0_test, xTest] = generate_trajectories(num_test, t, params);
    
    % Define model variants for ablation study - using cell array with structs
    model_names = {'Full Physics-Informed', 'No Momentum Loss', 'No Position Constraint', 'No Physics (Vanilla)'};
    model_configs = cell(4, 1);
    
    % Model 1: Full Physics-Informed
    model_configs{1} = struct('use_pos_constraint', true, 'use_momentum_loss', true);
    
    % Model 2: No Momentum Loss
    model_configs{2} = struct('use_pos_constraint', true, 'use_momentum_loss', false);
    
    % Model 3: No Position Constraint
    model_configs{3} = struct('use_pos_constraint', false, 'use_momentum_loss', true);
    
    % Model 4: No Physics (Vanilla)
    model_configs{4} = struct('use_pos_constraint', false, 'use_momentum_loss', false);
    
    % Initialize results storage
    num_models = length(model_names);
    ablation_results = struct();
    ablation_results.model_names = model_names;
    ablation_results.train_losses = cell(num_models, 1);
    ablation_results.test_errors = zeros(num_models, 1);
    ablation_results.conservation_errors = zeros(num_models, 1);
    ablation_results.long_term_errors = zeros(num_models, 1);
    
    % Train and evaluate each model variant
    for m = 1:num_models
        model_name = model_names{m};
        model_config = model_configs{m};
        
        fprintf('\n=== Training %s Model ===\n', model_name);
        
        % Initialize network with same architecture for all variants
        net_params = initialize_network(10, hidden_size);
        
        % Set up training parameters
        train_params = struct();
        train_params.num_iter = 3000;
        train_params.batch_size = 32;
        train_params.seq_length = 50;
        train_params.learning_rate = learning_rate;
        train_params.plot_freq = 500;
        train_params.momentum_weight = momentum_weight;
        train_params.use_pos_constraint = model_config.use_pos_constraint;
        train_params.use_momentum_loss = model_config.use_momentum_loss;
        
        % Train model
        [trained_model, losses] = train_ablation_model(net_params, xTrain, t, train_params);
        ablation_results.train_losses{m} = losses;
        
        % Evaluate on test set
        test_error = evaluate_ablation_model(trained_model, xTest, x0_test, t, model_config);
        ablation_results.test_errors(m) = test_error;
        fprintf('Test Error: %.6f\n', test_error);
        
        % Evaluate conservation laws
        conservation_error = evaluate_conservation(trained_model, xTest, x0_test, t, params, model_config);
        ablation_results.conservation_errors(m) = conservation_error;
        fprintf('Conservation Error: %.6f\n', conservation_error);
        
        % Evaluate long-term stability
        long_term_error = evaluate_long_term(trained_model, x0_test(:,1:5), t, params, model_config);
        ablation_results.long_term_errors(m) = long_term_error;
        fprintf('Long-term Error: %.6f\n', long_term_error);
        
        % Save model
        save(sprintf('ablation_model_%d.mat', m), 'trained_model', 'model_name', 'model_config');
    end
    
    % Create visualization of ablation results
    visualize_ablation_results(ablation_results);
    
    % Save results
    save('ablation_results.mat', 'ablation_results');
    fprintf('\nAblation study complete. Results saved to ablation_results.mat\n');
end

function [trained_model, losses] = train_ablation_model(net_params, xTrain, t, train_params)
    % Initialize storage for losses
    losses = zeros(train_params.num_iter, 1);
    
    % Initialize Adam optimizer parameters
    averageGrad = [];
    averageSqGrad = [];
    gradDecay = 0.9;
    sqGradDecay = 0.999;
    
    % Create figure for training visualization
    figure('Name', sprintf('Training Progress - %s', ...
           get_model_description(train_params)));
    
    % Training loop
    for iteration = 1:train_params.num_iter
        % Create batch
        [X0, targets] = create_batch(xTrain, train_params.batch_size, train_params.seq_length);
        
        % Compute loss and gradients based on ablation configuration
        [loss, gradients] = dlfeval(@(t, X0, net_params, targets, params) compute_ablation_loss(t, X0, net_params, targets, params), ...
                                  t(1:train_params.seq_length), X0, net_params, targets, train_params);
        
        % Update parameters
        [net_params, averageGrad, averageSqGrad] = adamupdate(net_params, gradients, ...
            averageGrad, averageSqGrad, iteration, train_params.learning_rate, gradDecay, sqGradDecay);
        
        % Store loss
        losses(iteration) = double(gather(extractdata(loss)));
        
        % Display progress
        if mod(iteration, train_params.plot_freq) == 0
            fprintf('Iteration: %d, Loss: %.4f\n', iteration, losses(iteration));
            
            % Update training progress plot
            plot(1:iteration, losses(1:iteration), 'b-', 'LineWidth', 1.5);
            title(sprintf('Training Loss - %s', get_model_description(train_params)));
            xlabel('Iteration');
            ylabel('Loss');
            grid on;
            drawnow;
        end
    end
    
    % Return trained model
    trained_model = net_params;
end

function [loss, gradients] = compute_ablation_loss(t, X0, net_params, targets, params)
    % Forward pass using appropriate ODE function based on configuration
    if params.use_pos_constraint
        predictions = neural_ode_forward_ablation(t, X0, net_params, params.use_pos_constraint);
    else
        predictions = vanilla_neural_ode_forward(t, X0, net_params);
    end
    
    % Position and velocity components of the loss
    pos_pred = predictions(1:5,:,:);
    vel_pred = predictions(6:10,:,:);
    pos_true = targets(1:5,:,:);
    vel_true = targets(6:10,:,:);
    
    % Position and velocity loss
    pos_loss = mean(abs(pos_pred - pos_true), 'all');
    vel_loss = mean(abs(vel_pred - vel_true), 'all');
    
    % Combined loss components
    if params.use_momentum_loss
        % Include momentum conservation loss
        momentum_loss = compute_momentum_loss(predictions);
        loss = pos_loss + 2*vel_loss + params.momentum_weight*momentum_loss;
    else
        % Only state prediction loss
        loss = pos_loss + 2*vel_loss;
    end
    
    % Compute gradients
    gradients = dlgradient(loss, net_params);
end

function predictions = neural_ode_forward_ablation(t, X0, net_params, use_pos_constraint)
    tspan = [t(1), t(end)];
    
    if use_pos_constraint
        % Use physics-constrained ODE function
        predictions = dlode45(@ode_function_ablation, tspan, X0, net_params, 'DataFormat', "CB");
    else
        % Use vanilla ODE function
        predictions = dlode45(@vanilla_ode_function, tspan, X0, net_params, 'DataFormat', "CB");
    end
end

function predictions = vanilla_neural_ode_forward(t, X0, net_params)
    tspan = [t(1), t(end)];
    predictions = dlode45(@vanilla_ode_function, tspan, X0, net_params, 'DataFormat', "CB");
end

function dx = ode_function_ablation(t, x, params)
    % Neural network layers
    h1 = tanh(params.fc1.Weights * x + params.fc1.Bias);
    h2 = tanh(params.fc2.Weights * h1 + params.fc2.Bias);
    h3 = tanh(params.fc3.Weights * h2 + params.fc3.Bias);
    dx = params.fc4.Weights * h3 + params.fc4.Bias;
    
    % Physics-informed constraint: position derivatives are velocities
    dx(1:5) = x(6:10);
    
    % Scale acceleration terms as in the original code
    dx(6:7) = dx(6:7) * 0.01;     % Base linear accelerations
    dx(8) = dx(8) * 0.005;        % Base angular acceleration
    dx(9:10) = dx(9:10) * 0.005;  % Joint accelerations
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

function model_description = get_model_description(params)
    if params.use_pos_constraint && params.use_momentum_loss
        model_description = 'Full Physics-Informed';
    elseif params.use_pos_constraint && ~params.use_momentum_loss
        model_description = 'No Momentum Loss';
    elseif ~params.use_pos_constraint && params.use_momentum_loss
        model_description = 'No Position Constraint';
    else
        model_description = 'No Physics (Vanilla)';
    end
end

function test_error = evaluate_ablation_model(model, xTest, x0_test, t, config)
    num_test = size(x0_test, 2);
    total_error = 0;
    
    % Evaluate on each test trajectory
    for i = 1:num_test
        % Get ground truth
        x_true = xTest{i};
        
        % Predict trajectory using appropriate function based on configuration
        if config.use_pos_constraint
            x_pred = predict_trajectory_ablation(t, x0_test(:,i), model, config);
        else
            x_pred = predict_vanilla_trajectory(t, x0_test(:,i), model);
        end
        
        % Calculate normalized error
        error = mean(vecnorm(x_pred - x_true, 2, 1) ./ vecnorm(x_true, 2, 1));
        total_error = total_error + error;
    end
    
    % Average error across test set
    test_error = total_error / num_test;
end

function x_pred = predict_trajectory_ablation(t, x0, net_params, config)
    n_steps = length(t);
    x_pred = zeros(10, n_steps);
    x_pred(:,1) = x0;
    
    dt = t(2) - t(1);
    for i = 1:n_steps-1
        tspan = [0, dt];
        current_state = dlarray(x_pred(:,i));
        
        % Use appropriate ODE function based on configuration
        if config.use_pos_constraint
            next_state = dlode45(@ode_function_ablation, tspan, current_state, net_params, 'DataFormat', "CB");
        else
            next_state = dlode45(@vanilla_ode_function, tspan, current_state, net_params, 'DataFormat', "CB");
        end
        
        x_pred(:,i+1) = extractdata(next_state(:,end));
    end
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

function conservation_error = evaluate_conservation(model, xTest, x0_test, t, params, config)
    num_test = size(x0_test, 2);
    lin_mom_error = 0;
    ang_mom_error = 0;
    
    % Evaluate conservation laws on test trajectories
    for i = 1:num_test
        % Predict trajectory
        if config.use_pos_constraint
            x_pred = predict_trajectory_ablation(t, x0_test(:,i), model, config);
        else
            x_pred = predict_vanilla_trajectory(t, x0_test(:,i), model);
        end
        
        % Calculate initial momenta
        lin_mom_init = calculate_linear_momentum(x_pred(:,1), params);
        ang_mom_init = calculate_angular_momentum(x_pred(:,1), params);
        
        % Calculate conservation errors
        lin_mom_deviation = 0;
        ang_mom_deviation = 0;
        
        for j = 1:length(t)
            lin_mom_j = calculate_linear_momentum(x_pred(:,j), params);
            ang_mom_j = calculate_angular_momentum(x_pred(:,j), params);
            
            lin_mom_deviation = lin_mom_deviation + abs(lin_mom_j - lin_mom_init) / lin_mom_init;
            ang_mom_deviation = ang_mom_deviation + abs(ang_mom_j - ang_mom_init) / ang_mom_init;
        end
        
        % Average deviations over time
        lin_mom_error = lin_mom_error + lin_mom_deviation / length(t);
        ang_mom_error = ang_mom_error + ang_mom_deviation / length(t);
    end
    
    % Combined conservation error (average over test set)
    conservation_error = (lin_mom_error + ang_mom_error) / (2 * num_test);
end

function long_term_error = evaluate_long_term(model, x0_test, t, params, config)
    % Extended simulation time
    T_long = 30;  % 30 seconds (3x longer than training)
    dt = 0.01;
    t_long = 0:dt:T_long;
    
    num_test = size(x0_test, 2);
    error_accum = 0;
    
    for i = 1:num_test
        % Generate ground truth for longer horizon
        [~, x_true_long] = ode45(@(t,x) FFSM_dynamics(t, x, zeros(2,1), params), t_long, x0_test(:,i));
        x_true_long = x_true_long';
        
        % Predict long-term trajectory using appropriate function
        if config.use_pos_constraint
            x_pred_long = predict_trajectory_ablation(t_long, x0_test(:,i), model, config);
        else
            x_pred_long = predict_vanilla_trajectory(t_long, x0_test(:,i), model);
        end
        
        % Calculate error focusing on the latter half of the trajectory
        half_idx = floor(length(t_long) / 2);
        long_term_segment_true = x_true_long(:, half_idx:end);
        long_term_segment_pred = x_pred_long(:, half_idx:end);
        
        % Normalized error for long-term prediction
        error = mean(vecnorm(long_term_segment_pred - long_term_segment_true, 2, 1) ./ ...
                     vecnorm(long_term_segment_true, 2, 1));
        
        error_accum = error_accum + error;
    end
    
    % Average long-term prediction error
    long_term_error = error_accum / num_test;
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

function visualize_ablation_results(results)
    % Create main figure
    figure('Name', 'Ablation Study Results', 'Position', [100 100 1200 800]);
    
    % 1. Test error comparison
    subplot(2, 2, 1);
    bar(results.test_errors);
    set(gca, 'XTickLabel', results.model_names);
    title('Test Error Comparison');
    ylabel('Normalized Error');
    grid on;
    
    % 2. Conservation law adherence
    subplot(2, 2, 2);
    bar(results.conservation_errors);
    set(gca, 'XTickLabel', results.model_names);
    title('Conservation Law Error');
    ylabel('Conservation Error');
    grid on;
    
    % 3. Long-term stability
    subplot(2, 2, 3);
    bar(results.long_term_errors);
    set(gca, 'XTickLabel', results.model_names);
    title('Long-term Prediction Error');
    ylabel('Normalized Error');
    grid on;
    
    % 4. Training loss curves
    subplot(2, 2, 4);
    hold on;
    colors = {'b', 'r', 'g', 'm'};
    for i = 1:length(results.model_names)
        plot(results.train_losses{i}, colors{i}, 'LineWidth', 1.5);
    end
    hold off;
    title('Training Loss Comparison');
    xlabel('Iteration');
    ylabel('Loss');
    legend(results.model_names, 'Location', 'northeast');
    grid on;
    
    % Save figure
    saveas(gcf, 'ablation_study_results.png');
    
    % Create detailed comparison figure for each metric
    figure('Name', 'Ablation Study Component Analysis', 'Position', [100 100 800 800]);
    
    % Calculate impact of each component
    baseline = results.test_errors(4);  % Vanilla model
    full_model = results.test_errors(1);  % Full physics-informed
    pos_impact = results.test_errors(3) - results.test_errors(1);  % Adding position constraint
    mom_impact = results.test_errors(2) - results.test_errors(1);  % Adding momentum loss
    
    % Create waterfall chart
    subplot(2, 1, 1);
    data = [baseline, -pos_impact, -mom_impact, full_model];
    bar_labels = {'Vanilla', 'Position\nConstraint', 'Momentum\nLoss', 'Full\nModel'};
    
    % Plot waterfall
    colors = [0.8 0.2 0.2; 0.2 0.6 0.2; 0.2 0.2 0.8; 0.2 0.8 0.8];
    bar(1:4, data, 'FaceColor', 'flat');
    for i = 1:4
        b = bar(i, data(i));
        b.FaceColor = colors(i,:);
    end
    
    title('Component Contribution to Error Reduction');
    set(gca, 'XTickLabel', bar_labels);
    ylabel('Error Contribution');
    grid on;
    
    % Summary table
    subplot(2, 1, 2);
    axis off;
    
    % Calculate percentage improvements
    pos_pct = (pos_impact / baseline) * 100;
    mom_pct = (mom_impact / baseline) * 100;
    total_pct = ((baseline - full_model) / baseline) * 100;
    
    % Create text summary
    text(0.1, 0.9, 'Ablation Study Summary:', 'FontWeight', 'bold', 'FontSize', 14);
    text(0.1, 0.8, sprintf('Baseline (Vanilla) Error: %.6f', baseline));
    text(0.1, 0.7, sprintf('Full Physics-Informed Error: %.6f', full_model));
    text(0.1, 0.6, sprintf('Error Reduction: %.6f (%.2f%%)', baseline - full_model, total_pct));
    text(0.1, 0.5, 'Component Contributions:', 'FontWeight', 'bold');
    text(0.1, 0.4, sprintf('Position Constraint: %.6f (%.2f%%)', pos_impact, pos_pct));
    text(0.1, 0.3, sprintf('Momentum Conservation: %.6f (%.2f%%)', mom_impact, mom_pct));
    
    % Concluding remarks
    if pos_impact > mom_impact
        text(0.1, 0.2, 'The Position Constraint provides the most significant improvement.', 'FontWeight', 'bold');
    else
        text(0.1, 0.2, 'The Momentum Conservation Loss provides the most significant improvement.', 'FontWeight', 'bold');
    end
    
    if total_pct > (pos_pct + mom_pct)
        text(0.1, 0.1, 'The components show synergistic effects when combined.', 'FontWeight', 'bold');
    elseif total_pct < (pos_pct + mom_pct)
        text(0.1, 0.1, 'The components show redundant effects when combined.', 'FontWeight', 'bold');
    else
        text(0.1, 0.1, 'The components contribute additively to the improvement.', 'FontWeight', 'bold');
    end
    
    % Save figure
    saveas(gcf, 'ablation_component_analysis.png');
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









function plot_training_progress(losses)
    figure('Name', 'Final Training Progress');
    plot(losses, 'b-', 'LineWidth', 1.5);
    title('Training Loss History');
    xlabel('Iteration');
    ylabel('Loss');
    grid on;
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
