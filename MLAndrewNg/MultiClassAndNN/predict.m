function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

X = [ones(m, 1) X]';
A_2 = [ones(1, m); sigmoid(Theta1 * X)];
A_3 = sigmoid(Theta2 * A_2)';
[ma p] = max(A_3, [], 2);

end
