function B = sigmoidGradient(A)

B = sigmoid(A).*(1-sigmoid(A));

end
