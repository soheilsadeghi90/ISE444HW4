function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size)), ...
                 hidden_layer_size, (input_layer_size));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size))):end), ...
                 num_labels, (hidden_layer_size));

m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
% Theta1_grad = zeros(size(Theta1));
% Theta2_grad = zeros(size(Theta2));
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. 
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. 
%
% Part 3: Implement regularization with the cost function and gradients.
%
%

a_1 = X;
z_2 = a_1*Theta1';
a_2 = sigmoid(z_2);
% a_2 = [ones(m,1) a_2];
z_3 = a_2*Theta2';
h = sigmoid(z_3);
a_3 = h; 

% step 1 Feedforward and cost function
for i = 1:m
    J = J + (-1/m)*(log(h(i,:))*y(i,:)'+log(1-h(i,:))*(1-y(i,:))');
end
temp1 = Theta1;
temp1(:,1) = 0;
temp2 = Theta2;
temp2(:,1) = 0;
J = J + (lambda/2/m)*(sum(sum(temp1.^2)) + sum(sum(temp2.^2)));

% step 2 Backpropagation and gradient

delta_1 = zeros(input_layer_size , hidden_layer_size);
delta_2 = zeros(hidden_layer_size , num_labels);

for t = 1 : m
    del_3 = a_3(t,:) - y(t,:);
    sig = sigmoidGradient(z_2(t,:));
    sig(:,1) = 1;
    del_2 = del_3*temp2.*sig;
    delta_1 = delta_1 + a_1(t,:)'*del_2;
    delta_2 = delta_2 + a_2(t,:)'*del_3;
end

Theta1_grad = (1/m)*delta_1' + (lambda/m)*Theta1;
Theta1_grad(:,1) = (1/m)*delta_1(1,:)';
Theta2_grad = (1/m)*delta_2' + (lambda/m)*Theta2;
Theta2_grad(:,1) = (1/m)*delta_2(1,:)';



% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end