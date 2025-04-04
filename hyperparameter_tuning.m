function hyperparameter_tuning()
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
    
    % Generate training and validation data
    num_train = 80;
    num_val = 20;
    fprintf('Generating %d training and %d validation trajectories...\n', num_train, num_val);
    [x0_train, xTrain] = generate_trajectories(num_train, t, params);
    [x0_val, xVal] = generate_trajectories(num_val, t, params);
    
    % Hyperparameter grid search for physics-informed model
    fprintf('\nHyperparameter tuning for Physics-Informed Neural ODE\n');
    fprintf('=====================================================\n');
    physics_results = physics_informed_hyperparameter_tuning(xTrain, xVal, x0_val, t, params);
    
    % Hyperparameter grid search for vanilla model
    fprintf('\nHyperparameter tuning for Vanilla Neural ODE\n');
    fprintf('============================================\n');
    vanilla_results = vanilla_hyperparameter_tuning(xTrain, xVal, x0_val, t, params);
    
    % Plot results
    plot_hyperparameter_results(physics_results, vanilla_results);
    
    % Save results
    save('hyperparameter_tuning_results.mat', 'physics_results', 'vanilla_results');
    fprintf('\nHyperparameter tuning complete. Results saved to hyperparameter_tuning_results.mat\n');
end














function physics_results = physics_informed_hyperparameter_tuning(xTrain, xVal, x0_val, t, params)
    % Hyperparameter grid
    hidden_sizes = [128, 256, 512]; 
    learning_rates = [0.0001, 0.0005, 0.001];
    momentum_weights = [1, 5, 10];  % Weight for momentum conservation loss
    
    % Initialize results storage
    num_configs = length(hidden_sizes) * length(learning_rates) * length(momentum_weights);
    physics_results = struct();
    physics_results.hidden_sizes = hidden_sizes;
    physics_results.learning_rates = learning_rates;
    physics_results.momentum_weights = momentum_weights;
    physics_results.val_errors = zeros(length(hidden_sizes), length(learning_rates), length(momentum_weights));
    physics_results.train_losses = cell(length(hidden_sizes), length(learning_rates), length(momentum_weights));
    physics_results.best_model = [];
    physics_results.best_error = inf;
    physics_results.best_config = [];
    
    % Keep track of validation errors for plotting
    plotValues = zeros(num_configs, 1);
    
    % Initialize figure for tracking progress
    figure('Name', 'Physics-Informed Hyperparameter Tuning Progress');
    
    % Run grid search
    config_num = 0;
    for h = 1:length(hidden_sizes)
        hidden_size = hidden_sizes(h);
        
        for l = 1:length(learning_rates)
            learning_rate = learning_rates(l);
            
            for m = 1:length(momentum_weights)
                momentum_weight = momentum_weights(m);
                config_num = config_num + 1;
                
                fprintf('\nTesting config %d/%d: hidden_size = %d, lr = %.5f, momentum_weight = %.1f\n', ...
                        config_num, num_configs, hidden_size, learning_rate, momentum_weight);
                
                % Initialize network
                net_params = initialize_network(10, hidden_size);
                
                % Set up training parameters
                train_params = struct();
                train_params.num_iter = 2000;  % Reduced for grid search
                train_params.batch_size = 32;
                train_params.seq_length = 50;
                train_params.learning_rate = learning_rate;
                train_params.plot_freq = 500;
                train_params.momentum_weight = momentum_weight;
                
                % Train model
                [trained_model, losses] = train_physics_model(net_params, xTrain, t, train_params);
                
                % Evaluate on validation set
                val_error = evaluate_physics_model(trained_model, xVal, x0_val, t);
                physics_results.val_errors(h, l, m) = val_error;
                physics_results.train_losses{h, l, m} = losses;
                plotValues(config_num) = val_error;  % Store for plotting
                
                fprintf('Validation Error: %.6f\n', val_error);
                
                % Update best model if needed
                if val_error < physics_results.best_error
                    physics_results.best_error = val_error;
                    physics_results.best_model = trained_model;
                    physics_results.best_config = [hidden_size, learning_rate, momentum_weight];
                    fprintf('New best model found!\n');
                end
                
                % Plot current progress
                subplot(2, 1, 1);
                plot(losses);
                title(sprintf('Training Loss - Config %d/%d', config_num, num_configs));
                xlabel('Iteration');
                ylabel('Loss');
                grid on;
                
                subplot(2, 1, 2);
                bar(1:config_num, plotValues(1:config_num));
                title('Validation Errors by Configuration');
                xlabel('Configuration Number');
                ylabel('Validation Error');
                grid on;
                
                drawnow;
            end
        end
    end
    
    % Save best model
    trained_model = physics_results.best_model;
    save('best_physics_model.mat', 'trained_model');
    
    % Display best configuration
    fprintf('\nBest Physics-Informed Configuration:\n');
    fprintf('Hidden Size: %d\n', physics_results.best_config(1));
    fprintf('Learning Rate: %.5f\n', physics_results.best_config(2));
    fprintf('Momentum Weight: %.1f\n', physics_results.best_config(3));
    fprintf('Validation Error: %.6f\n', physics_results.best_error);
    
    % Plot hyperparameter heatmaps
    plot_physics_heatmaps(physics_results);
end

function [trained_model, losses] = train_physics_model(net_params, xTrain, t, train_params)
    % Similar to train_model function but with momentum_weight as parameter
    
    % Initialize storage for losses
    losses = zeros(train_params.num_iter, 1);
    
    % Initialize Adam optimizer parameters
    averageGrad = [];
    averageSqGrad = [];
    gradDecay = 0.9;
    sqGradDecay = 0.999;
    
    % Training loop
    for iteration = 1:train_params.num_iter
        % Create batch
        [X0, targets] = create_batch(xTrain, train_params.batch_size, train_params.seq_length);
        
        % Compute loss and gradients with custom momentum weight
        [loss, gradients] = dlfeval(@(t, X0, net_params, targets, mw) compute_physics_loss(t, X0, net_params, targets, mw), ...
                                  t(1:train_params.seq_length), X0, net_params, targets, train_params.momentum_weight);
        
        % Update parameters
        [net_params, averageGrad, averageSqGrad] = adamupdate(net_params, gradients, ...
            averageGrad, averageSqGrad, iteration, train_params.learning_rate, gradDecay, sqGradDecay);
        
        % Store loss
        losses(iteration) = double(gather(extractdata(loss)));
        
        % Display progress
        if mod(iteration, train_params.plot_freq) == 0
            fprintf('Iteration: %d, Loss: %.4f\n', iteration, losses(iteration));
        end
    end
    
    % Return trained model
    trained_model = net_params;
end

function [loss, gradients] = compute_physics_loss(t, X0, net_params, targets, momentum_weight)
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
    
    % Combined loss with custom weight for momentum term
    loss = pos_loss + 2*vel_loss + momentum_weight*momentum_loss;
    
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

function val_error = evaluate_physics_model(model, xVal, x0_val, t)
    num_val = size(x0_val, 2);
    total_error = 0;
    
    % Evaluate on each validation trajectory
    for i = 1:num_val
        % Get ground truth
        x_true = xVal{i};
        
        % Predict trajectory
        x_pred = predict_trajectory(t, x0_val(:,i), model);
        
        % Calculate normalized error
        error = mean(vecnorm(x_pred - x_true, 2, 1) ./ vecnorm(x_true, 2, 1));
        total_error = total_error + error;
    end
    
    % Average error across validation set
    val_error = total_error / num_val;
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

function plot_physics_heatmaps(physics_results)
    % Create figure for heatmaps
    figure('Name', 'Physics-Informed Hyperparameter Heatmaps', 'Position', [100 100 1200 400]);
    
    % For each momentum weight, create a heatmap of hidden size vs learning rate
    for m = 1:length(physics_results.momentum_weights)
        subplot(1, length(physics_results.momentum_weights), m);
        
        % Extract data for this momentum weight
        error_slice = physics_results.val_errors(:,:,m);
        
        % Create heatmap
        imagesc(error_slice);
        colormap(jet);
        colorbar;
        
        % Set axis labels
        set(gca, 'XTick', 1:length(physics_results.learning_rates));
        set(gca, 'XTickLabel', physics_results.learning_rates);
        set(gca, 'YTick', 1:length(physics_results.hidden_sizes));
        set(gca, 'YTickLabel', physics_results.hidden_sizes);
        
        title(sprintf('Momentum Weight = %.1f', physics_results.momentum_weights(m)));
        xlabel('Learning Rate');
        ylabel('Hidden Size');
    end
    
    saveas(gcf, 'physics_hyperparameter_heatmaps.png');
end


function vanilla_results = vanilla_hyperparameter_tuning(xTrain, xVal, x0_val, t, params)
    % Hyperparameter grid
    hidden_sizes = [128, 256, 512]; 
    learning_rates = [0.0001, 0.0005, 0.001];
    batch_sizes = [16, 32, 64];
    
    % Initialize results storage
    num_configs = length(hidden_sizes) * length(learning_rates) * length(batch_sizes);
    vanilla_results = struct();
    vanilla_results.hidden_sizes = hidden_sizes;
    vanilla_results.learning_rates = learning_rates;
    vanilla_results.batch_sizes = batch_sizes;
    vanilla_results.val_errors = zeros(length(hidden_sizes), length(learning_rates), length(batch_sizes));
    vanilla_results.train_losses = cell(length(hidden_sizes), length(learning_rates), length(batch_sizes));
    vanilla_results.best_model = [];
    vanilla_results.best_error = inf;
    vanilla_results.best_config = [];
    
    % Keep track of validation errors for plotting
    plotValues = zeros(num_configs, 1);
    
    % Initialize figure for tracking progress
    figure('Name', 'Vanilla Neural ODE Hyperparameter Tuning Progress');
    
    % Run grid search
    config_num = 0;
    for h = 1:length(hidden_sizes)
        hidden_size = hidden_sizes(h);
        
        for l = 1:length(learning_rates)
            learning_rate = learning_rates(l);
            
            for b = 1:length(batch_sizes)
                batch_size = batch_sizes(b);
                config_num = config_num + 1;
                
                fprintf('\nTesting config %d/%d: hidden_size = %d, lr = %.5f, batch_size = %d\n', ...
                        config_num, num_configs, hidden_size, learning_rate, batch_size);
                
                % Initialize network
                net_params = initialize_network(10, hidden_size);
                
                % Set up training parameters
                train_params = struct();
                train_params.num_iter = 2000;  % Reduced for grid search
                train_params.batch_size = batch_size;
                train_params.seq_length = 50;
                train_params.learning_rate = learning_rate;
                train_params.plot_freq = 500;
                
                % Train model
                [trained_model, losses] = train_vanilla_model(net_params, xTrain, t, train_params);
                
                % Evaluate on validation set
                val_error = evaluate_vanilla_model(trained_model, xVal, x0_val, t);
                vanilla_results.val_errors(h, l, b) = val_error;
                vanilla_results.train_losses{h, l, b} = losses;
                plotValues(config_num) = val_error;  % Store for plotting
                
                fprintf('Validation Error: %.6f\n', val_error);
                
                % Update best model if needed
                if val_error < vanilla_results.best_error
                    vanilla_results.best_error = val_error;
                    vanilla_results.best_model = trained_model;
                    vanilla_results.best_config = [hidden_size, learning_rate, batch_size];
                    fprintf('New best model found!\n');
                end
                
                % Plot current progress
                subplot(2, 1, 1);
                plot(losses);
                title(sprintf('Training Loss - Config %d/%d', config_num, num_configs));
                xlabel('Iteration');
                ylabel('Loss');
                grid on;
                
                subplot(2, 1, 2);
                bar(1:config_num, plotValues(1:config_num));
                title('Validation Errors by Configuration');
                xlabel('Configuration Number');
                ylabel('Validation Error');
                grid on;
                
                drawnow;
            end
        end
    end
    
    % Save best model
    trained_model = vanilla_results.best_model;
    save('best_vanilla_model.mat', 'trained_model');
    
    % Display best configuration
    fprintf('\nBest Vanilla Configuration:\n');
    fprintf('Hidden Size: %d\n', vanilla_results.best_config(1));
    fprintf('Learning Rate: %.5f\n', vanilla_results.best_config(2));
    fprintf('Batch Size: %d\n', vanilla_results.best_config(3));
    fprintf('Validation Error: %.6f\n', vanilla_results.best_error);
    
    % Plot hyperparameter heatmaps
    plot_vanilla_heatmaps(vanilla_results);
end

function [trained_model, losses] = train_vanilla_model(net_params, xTrain, t, train_params)
    % Initialize storage for losses
    losses = zeros(train_params.num_iter, 1);
    
    % Initialize Adam optimizer parameters
    averageGrad = [];
    averageSqGrad = [];
    gradDecay = 0.9;
    sqGradDecay = 0.999;
    
    % Training loop
    for iteration = 1:train_params.num_iter
        % Create batch
        [X0, targets] = create_batch(xTrain, train_params.batch_size, train_params.seq_length);
        
        % Compute loss and gradients - using vanilla loss function
        [loss, gradients] = dlfeval(@compute_vanilla_loss, t(1:train_params.seq_length), X0, net_params, targets);
        
        % Update parameters
        [net_params, averageGrad, averageSqGrad] = adamupdate(net_params, gradients, ...
            averageGrad, averageSqGrad, iteration, train_params.learning_rate, gradDecay, sqGradDecay);
        
        % Store loss
        losses(iteration) = double(gather(extractdata(loss)));
        
        % Display progress
        if mod(iteration, train_params.plot_freq) == 0
            fprintf('Iteration: %d, Loss: %.4f\n', iteration, losses(iteration));
        end
    end
    
    % Return trained model
    trained_model = net_params;
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

function val_error = evaluate_vanilla_model(model, xVal, x0_val, t)
    num_val = size(x0_val, 2);
    total_error = 0;
    
    % Evaluate on each validation trajectory
    for i = 1:num_val
        % Get ground truth
        x_true = xVal{i};
        
        % Predict trajectory
        x_pred = predict_vanilla_trajectory(t, x0_val(:,i), model);
        
        % Calculate normalized error
        error = mean(vecnorm(x_pred - x_true, 2, 1) ./ vecnorm(x_true, 2, 1));
        total_error = total_error + error;
    end
    
    % Average error across validation set
    val_error = total_error / num_val;
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

function plot_vanilla_heatmaps(vanilla_results)
    % Create figure for heatmaps
    figure('Name', 'Vanilla Neural ODE Hyperparameter Heatmaps', 'Position', [100 100 1200 400]);
    
    % For each batch size, create a heatmap of hidden size vs learning rate
    for b = 1:length(vanilla_results.batch_sizes)
        subplot(1, length(vanilla_results.batch_sizes), b);
        
        % Extract data for this batch size
        error_slice = vanilla_results.val_errors(:,:,b);
        
        % Create heatmap
        imagesc(error_slice);
        colormap(jet);
        colorbar;
        
        % Set axis labels
        set(gca, 'XTick', 1:length(vanilla_results.learning_rates));
        set(gca, 'XTickLabel', vanilla_results.learning_rates);
        set(gca, 'YTick', 1:length(vanilla_results.hidden_sizes));
        set(gca, 'YTickLabel', vanilla_results.hidden_sizes);
        
        title(sprintf('Batch Size = %d', vanilla_results.batch_sizes(b)));
        xlabel('Learning Rate');
        ylabel('Hidden Size');
    end
    
    saveas(gcf, 'vanilla_hyperparameter_heatmaps.png');
end






function plot_hyperparameter_results(physics_results, vanilla_results)
    % Create summary figure comparing best configurations
    figure('Name', 'Hyperparameter Tuning Summary', 'Position', [100 100 800 600]);
    
    % Bar plot comparing validation errors
    subplot(2, 1, 1);
    bar([physics_results.best_error, vanilla_results.best_error]);
    set(gca, 'XTickLabel', {'Physics-Informed', 'Vanilla NN'});
    title('Validation Error of Best Configurations');
    ylabel('Normalized Error');
    grid on;
    
    % Display configurations
    subplot(2, 1, 2);
    text(0.1, 0.9, 'Best Physics-Informed Configuration:', 'FontWeight', 'bold');
    text(0.1, 0.8, sprintf('Hidden Size: %d', physics_results.best_config(1)));
    text(0.1, 0.7, sprintf('Learning Rate: %.5f', physics_results.best_config(2)));
    text(0.1, 0.6, sprintf('Momentum Weight: %.1f', physics_results.best_config(3)));
    text(0.1, 0.5, sprintf('Validation Error: %.6f', physics_results.best_error));
    
    text(0.6, 0.9, 'Best Vanilla Configuration:', 'FontWeight', 'bold');
    text(0.6, 0.8, sprintf('Hidden Size: %d', vanilla_results.best_config(1)));
    text(0.6, 0.7, sprintf('Learning Rate: %.5f', vanilla_results.best_config(2)));
    text(0.6, 0.6, sprintf('Batch Size: %d', vanilla_results.best_config(3)));
    text(0.6, 0.5, sprintf('Validation Error: %.6f', vanilla_results.best_error));
    
    % Calculate improvement
    improvement = (vanilla_results.best_error - physics_results.best_error) / vanilla_results.best_error * 100;
    text(0.3, 0.3, sprintf('Physics-Informed Model Improvement: %.2f%%', improvement), 'FontWeight', 'bold');
    
    % Remove axes
    axis off;
    
    saveas(gcf, 'hyperparameter_tuning_summary.png');
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
