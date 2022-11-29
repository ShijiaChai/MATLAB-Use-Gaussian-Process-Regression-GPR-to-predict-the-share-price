close all;
clear all;
clc;
%% data
% y=f(X)+epsilon~N(0,sigma_N)
confidence_interval=0.95; 
sigma2_N = 0.001;
X = (1:0.5:5).';
n = length(X);
y = sin(X);
X_r=repmat(X,1,n);
X_r=X_r-X_r.';
X_r=X_r.^2;
A_RQ=1;
M_RQ=1;
K_RQ=2;
%% calc hyper-parameter
options=optimset('GradObj','on');
f_min=+inf;
init_guess=log([A_RQ;M_RQ;K_RQ]);
for i=1:20
    [hyper_parameter_possible,f]=fminunc(@(hyper_parameter)Kernel_RQ(X_r,y,sigma2_N,hyper_parameter),init_guess,options);
    if(f<f_min)
        f_min=f;
        hyper_parameter_log=hyper_parameter_possible;
    end
    init_guess=log(exp(hyper_parameter_log)+10*randn(length(hyper_parameter_log),1));
    while(any(imag(init_guess)))
        init_guess=log(exp(hyper_parameter_log)+10*randn(length(hyper_parameter_log),1));
    end
end
%% calc GP
hyper_parameter=exp(hyper_parameter_log);
A_RQ=hyper_parameter(1);
M_RQ=hyper_parameter(2);
K_RQ=hyper_parameter(3);
K=A_RQ*(1+M_RQ*X_r).^(-K_RQ)+sigma2_N*eye(n);
K_inv=inv(K);
X_star=(min(X)-2:0.1:max(X)+2).';
y_m=zeros(length(X_star),1);
y_sigma=zeros(length(X_star),1);
for i=1:length(X_star)
    K_star=A_RQ*(1+M_RQ*(repmat(X_star(i),n,1)-X).^2).^(-K_RQ);
    y_m(i)=K_star.'*K_inv*y;
    y_sigma(i)=sqrt(A_RQ+sigma2_N-K_star.'*K_inv*K_star);
end
%% plot
factor_sigma=qfuncinv((1-confidence_interval)/2);
figure;
hold on;
fill([X_star;flipud(X_star)],[y_m-factor_sigma*y_sigma;flipud(y_m+factor_sigma*y_sigma)],[248,195,205]/255);
plot(X,y,'*');
plot(X_star,y_m,'k','LineWidth',2);
hold off;