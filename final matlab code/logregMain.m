% DATABASE 4: mnist.mat (always run this section in advance, for all for four algorithms)

clc;
clear;

rng shuffle 

data_1 = load('new_data/mnist_mult.mat');

X_temp = data_1.TrainX;
Y_temp = data_1.TrainY;
Xt_temp = data_1.TestX;
Y_t_temp = data_1.TestY;

Y = zeros(size(Y_temp,1),1);
for i = 1:size(Y,1)
    Y(i) = find(Y_temp(i,:)==1);
end
Y(Y~=5) = -1;
Y(Y==5) = 1;

Y_t = zeros(size(Y_t_temp,1),1);
for i = 1:size(Y_t,1)
    Y_t(i) = find(Y_t_temp(i,:)==1);
end
Y_t(Y_t~=5) = -1;
Y_t(Y_t==5) = 1;

X = zeros(60000,28*28);
for i = 1:60000
    temp = squeeze(X_temp(:,:,i));
    X(i,:) = temp(:);
end

X_t = zeros(10000,28*28);
for i = 1:10000
    temp = squeeze(Xt_temp(:,:,i));
    X_t(i,:) = temp(:);
end

%% Scaling and parameters

for i = 1:size(X,1)
    X(i,:) = X(i,:)/norm(X(i,:),2);
end
for i = 1:size(X_t,1)
    X_t(i,:) = X_t(i,:)/norm(X_t(i,:),2);
end

w = rand(1,size(X,2));

lambda = 1/size(X,1);

f = objfnc(X,w,Y,lambda);
g = grad(X,w,Y,lambda);

batch_size = 128; % default = 50
epochs = 10;
iter = ceil(epochs*size(X,1)/batch_size);

%% solver (ADAM)

alfa = 0.002; % 0.002
beta_1 = 0.9;
beta_2 = 0.999;
eps = 1e-8;
m = zeros(1,numel(w));
v = zeros(1,numel(w));
t = 0;

gradi = 1;

J_tr_ADAM = zeros(iter,1);
J_ts_ADAM = zeros(iter,1);
w_ADAM = zeros(iter+1,numel(w));
w_ADAM(1,:) = w;

for i = 1:iter
    t = t + 1;
    Sk = randi(size(X,1),1,batch_size);
    gradi = grad(X(Sk,:),w_ADAM(i,:),Y(Sk,:),lambda);
    m = beta_1*m + (1-beta_1)*gradi;
    v = beta_2*v + (1-beta_2)*(gradi.^2);
    m_hat = m/(1-beta_1^t);
    v_hat = v/(1-beta_2^t);
    w_ADAM(i+1,:) = w_ADAM(i,:) - alfa * m_hat./(sqrt(v_hat)+eps);
    J_ts_ADAM(i) = objfnc(X_t,w_ADAM(i+1,:),Y_t,lambda);
    J_tr_ADAM(i) = objfnc(X,w_ADAM(i+1,:),Y,lambda);
%     obj_fnc = J_ts_ADAM(i)
%     norm(gradi,2)
    i
end

figure;plot(1:20:iter,[J_tr_ADAM(1:20:iter) J_ts_ADAM(1:20:iter)]);

%% solver (SGD)

a = 28350;
b = 30500;

k = 1:iter;
nu = a./(b+k);

J_tr_SGD = zeros(iter,1);
J_ts_SGD = zeros(iter,1);
w_SGD = zeros(iter+1,numel(w));
w_SGD(1,:) = w;

for j = 1:iter
    Sk = randi(size(X,1),1,batch_size);
    gradi = grad(X(Sk,:),w_SGD(j,:),Y(Sk,:),lambda);
    w_SGD(j+1,:) = w_SGD(j,:) - nu(j) * gradi;
    J_ts_SGD(j) = objfnc(X_t,w_SGD(j+1,:),Y_t,lambda);
    J_tr_SGD(j) = objfnc(X(Sk,:),w_SGD(j+1,:),Y(Sk,:),lambda);    
    j
end

figure;plot(1:20:iter,[J_tr_SGD(1:20:iter) J_ts_SGD(1:20:iter)]);

%% solver (SGD + Momentum)

a = 28350;
b = 30500;
teta = 0.95;

k = 1:iter;
nu = a./(b+k);

J_tr_SGDM = zeros(iter,1);
J_ts_SGDM = zeros(iter,1);
w_SGDM = zeros(iter+1,numel(w));
w_SGDM(1,:) = w;

v = zeros(1,length(w));
v_prev = zeros(1,length(w));

for j = 1:iter
    Sk = randi(size(X,1),1,batch_size);
    gradi = grad(X(Sk,:),w_SGDM(j,:),Y(Sk,:),lambda);
    v = teta * v_prev + gradi;      
    v_prev = v;
    w_SGDM(j+1,:) = w_SGDM(j,:) - nu(j) * v;
    J_ts_SGDM(j) = objfnc(X_t,w_SGDM(j+1,:),Y_t,lambda);
    J_tr_SGDM(j) = objfnc(X(Sk,:),w_SGDM(j+1,:),Y(Sk,:),lambda);    
    j
end

figure;plot(1:20:iter,[J_tr_SGDM(1:20:iter) J_ts_SGDM(1:20:iter)]);

%% Plots
step = 100;
% function in three methods
semilogy((1:step:iter)*batch_size/size(X,1),J_tr_SGD(1:step:end), '-.bs'); hold on
semilogy((1:step:iter)*batch_size/size(X,1),J_tr_SGDM(1:step:end), '--k+'); hold on
semilogy((1:step:iter)*batch_size/size(X,1),J_tr_ADAM(1:step:end), '-ro');
grid on
legend('SGD','SGD with Momentum','ADAM');
xlabel('epochs');
ylabel('objective function value');
title(['Logistic Regression / batch size = ',num2str(batch_size)])

% accuracy in three methods
pred_SGD = zeros(1,length(1:step:iter));
pred_SGDM = zeros(1,length(1:step:iter));
pred_ADAM = zeros(1,length(1:step:iter));
for i = 1:step:iter
    pred_SGD(i) = mean(double(predictlogreg(w_SGD(i+1,:), X) == Y));
    pred_SGDM(i) = mean(double(predictlogreg(w_SGDM(i+1,:), X) == Y));
    pred_ADAM(i) = mean(double(predictlogreg(w_ADAM(i+1,:), X) == Y));
end
figure;
plot((1:step:iter)*batch_size/size(X,1),pred_SGD(1:step:end), '-.bs'); hold on
plot((1:step:iter)*batch_size/size(X,1),pred_SGDM(1:step:end), '--k+'); hold on
plot((1:step:iter)*batch_size/size(X,1),pred_ADAM(1:step:end), '-ro');
grid on
legend('SGD','SGD with Momentum','ADAM');
xlabel('epochs');
ylabel('train accuracy value');
title(['Logistic Regression / batch size = ',num2str(batch_size)])

