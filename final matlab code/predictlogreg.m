function Y_pred = predictlogreg(w, X)
%PREDICT Predict the label of an input given a trained neural network

p = zeros(size(X, 1), 1);

Y_pred = w*X';
Y_pred(Y_pred>0) = 1;
Y_pred(Y_pred<0) = -1;
Y_pred = Y_pred';
% [dummy, p] = max(Y_pred, [], 2);

% =========================================================================


end
