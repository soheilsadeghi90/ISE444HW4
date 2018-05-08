function p = predict(nn_params, X,hidden_layer_size,input_layer_size,num_labels)
%PREDICT Predict the label of an input given a trained neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size)), ...
                 hidden_layer_size, (input_layer_size));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size))):end), ...
                 num_labels, (hidden_layer_size));
% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

% h1 = sigmoid([ones(m, 1) X] * Theta1');
% h2 = sigmoid([ones(m, 1) h1] * Theta2');
h1 = sigmoid(X * Theta1');
h2 = sigmoid(h1 * Theta2');
[dummy, p] = max(h2, [], 2);

% =========================================================================


end