function f = objfnc(X,w,Y,lambda)
m = size(X,1);
temp = log(1 + exp(-Y'.*(w*X')));
f = sum(temp)/m + lambda*(sum(w.*w))/2.0;
end