function J = nnCost(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% same as nnCostFunction, just does not return the gradient

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size)), ...
                 hidden_layer_size, (input_layer_size));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size))):end), ...
                 num_labels, (hidden_layer_size));
             
% Setup some useful variables
m = size(X, 1);
         
J = 0;
Y = y;
A1 = X;
Z2 = A1 * Theta1';
A2 = sigmoid(Z2);
Z3 = A2 * Theta2';
A3 = sigmoid(Z3);
H = A3;

% step 1 Feedforward and cost function
J = (1 / m) * sum(sum((-Y) .* log(H) - (1 - Y) .* log(1 - H), 2));

% step 2 Regularized cost function
penalty = (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end) .^ 2, 2)) + sum(sum(Theta2(:, 2:end) .^ 2, 2)));
J = J + penalty;


end