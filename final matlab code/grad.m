function g = grad(X,w,Y,lambda)
m = size(X,1);
temp = sum((1./(1 + exp(Y'.*(w*X')))).*(-Y'.*X'),2);
g = temp'/m + lambda*w;
end