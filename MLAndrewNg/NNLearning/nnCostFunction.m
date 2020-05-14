function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

m = size(X, 1);
X = [ones(m, 1) X]; % add bias units

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); 

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Backpropagation
for i = 1:m
    % Forward propagation
    a1 = X(i,:)';
    a2 = [1; sigmoid(Theta1 * a1)];
    a3 = sigmoid(Theta2 * a2);
    
    % Accumulate costs
    y_i = zeros(size(a3, 1), 1);
    y_i(y(i)) = 1;
    J = J - (y_i' * log(a3) + (1 - y_i)' * log(1 - a3));
    
    % Compute deltas
    d3 = a3 - y_i;
    d2 = (Theta2' * d3) .* (a2 .* (1 - a2));
    d2 = d2(2:end); % drop bias unit
    
    % Accumulate to gradient matrices
    Theta1_grad = Theta1_grad + (d2 * a1');
    Theta2_grad = Theta2_grad + (d3 * a2');
end

% Compute and regularize cost J
J = (1 / m) * J + (lambda / (2 * m)) * ...
(sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)));

% Compute gradients
Theta1_grad = (1 / m) * Theta1_grad;
Theta2_grad = (1 / m) * Theta2_grad;

% Regularization
Theta1_grad(:, 2:end) = ...
    Theta1_grad(:, 2:end) + (lambda / m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = ...
    Theta2_grad(:, 2:end) + (lambda / m) * Theta2(:, 2:end);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
