%% Prony Analysis using Signal Processing Toolbox
% This script uses MATLAB's built-in prony function (from the Signal Processing Toolbox)
% to perform Prony Analysis on your deck displacement data (z-values).
% It assumes the total simulation time is 73.343 seconds.

% Clear workspace and command window
clear; clc;

%% Step 1: Load Data and Generate Time Vector
% Replace 'ground_truth.csv' with your filename.
% It is assumed the CSV file contains a column named 'field.pose.pose.position.z'
data = readtable('ground_truth.csv');
y = data.('field_pose_pose_position_z');  % Measured deck displacement (z-values)
N = length(y);

total_time = 73.343;  % total simulation time in seconds
% Create a time vector uniformly spanning from 0 to total_time.
t = linspace(0, total_time, N)';
% Compute the effective sampling period.
T = t(2) - t(1);

fprintf('Loaded %d samples.\nTotal simulation time: %.3f s, Sampling period: %.4f s\n', N, total_time, T);

%% Step 2: Center the Data (Subtract the Mean)
y_mean = mean(y);
y_centered = y - y_mean;

%% Step 3: Choose Model Order and Perform Prony Analysis
% Set the model order (number of exponentials). If overfitting occurs, try a lower order.
n = 4;  % Adjust as needed (try 2, 3, or 4)
% Use MATLAB's built-in prony function to fit the centered data.
[B, A] = prony(y_centered, n, n);
fprintf('Model order used: %d\n', n);

%% Step 4: Compute Discrete-Time and Continuous-Time Poles
% Compute discrete-time poles (z-values) from the denominator polynomial A.
z = roots(A);
% Convert these discrete-time poles to continuous-time poles.
lambda = log(z) / T;
fprintf('Discrete-time poles (z):\n'); disp(z);
fprintf('Continuous-time poles (lambda):\n'); disp(lambda);

%% Step 5: Compute Residues using the residue Function
% The residue function performs partial fraction expansion.
[r, p, k] = residue(B, A);
fprintf('Residues (r):\n'); disp(r);
fprintf('Direct polynomial term (k):\n'); disp(k);
% Option: If k is nonzero (i.e., not negligible), it might introduce a large offset.
if ~isempty(k) && any(abs(k) > 1e-6)
    fprintf('Nonzero direct term detected. Omitting the direct term from reconstruction.\n');
    k = [];  % omit direct term for now
end

%% Step 6: Reconstruct the Full Signal Using the Prony Model
y_hat_centered = zeros(N,1);
for i = 1:length(r)
    y_hat_centered = y_hat_centered + r(i) * exp(lambda(i) * t);
end
if ~isempty(k)
    y_hat_centered = y_hat_centered + polyval(k, t);
end
% Add the mean back to obtain the final reconstructed signal.
y_hat = y_hat_centered + y_mean;

%% Step 7: Extract the Dominant Trend
% For a stable deck, we expect the dominant (slow-varying) trend to be represented by modes with negative real parts.
% We select only those modes with real(lambda) < -1e-4 (you can adjust this threshold).
dom_indices = find(real(lambda) < -1e-4);
if isempty(dom_indices)
    fprintf('No modes with sufficiently negative real parts found. Using all modes for the dominant trend.\n');
    dom_indices = 1:length(lambda);
end

dominant_trend_centered = zeros(N,1);
for i = dom_indices'
    dominant_trend_centered = dominant_trend_centered + r(i) * exp(lambda(i) * t);
end
% Add the mean back to the dominant trend.
dominant_trend = dominant_trend_centered + y_mean;

%% Step 8: Plot the Results Clearly
figure
hold on;
% Plot measured data as green circles.
plot(t, y, 'go', 'MarkerSize', 3, 'DisplayName', 'Measured z Position');
% Plot full Prony reconstruction as a red solid line.
plot(t, y_hat, 'r-', 'LineWidth', 2, 'DisplayName', 'Prony Reconstruction');
% Plot dominant trend as a blue dashed line.
plot(t, dominant_trend, 'b--', 'LineWidth', 2, 'DisplayName', 'Dominant Trend');
xlabel('Time (s)', 'FontSize', 12);
ylabel('Deck Displacement (units)', 'FontSize', 12);
title(sprintf('Prony Analysis of Deck Displacement Data\nTotal Duration = %.3f s, Model Order = %d', total_time, n), 'FontSize', 14);
legend('Location', 'best');
grid on;
hold off;