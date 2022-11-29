clear all;
clc;
%% data
confidence_interval=0.95;
data_xls=xlsread('data5.xls');
n=700;
m=20;
y=data_xls(4:n+3,4);
X=zeros(n,4);
X(:,1)=(3:n+2).';
X(:,2)=data_xls(3:n+2,4);
X(:,3)=data_xls(2:n+1,4);
X(:,4)=data_xls(3:n+2,5);
y_true=data_xls(n+4:n+m+3,4);
X_star=zeros(m,4);
X_star(:,1)=(n+3:n+m+2).';
X_star(:,2)=data_xls(n+3:n+m+2,4);
X_star(:,3)=data_xls(n+2:n+m+1,4);
X_star(:,4)=data_xls(n+3:n+m+2,5);
treatment=1;
%% pre-treatment
if(treatment==1)
    y_avg=mean(y);
    y=y-y_avg;
    y_true=y_true-y_avg;
elseif(treatment==2)
    X=X(2:end);
    y_0=y(1);
    y_true=y_true-[y(end);y_true(1:end-1)];
    y=y(2:end)-y(1:end-1);
elseif(treatment==3)
    for i=2:n
        y(i)=y(i)+y(i-1);
    end
    y_true(1)=y_true(1)+y(end);
    for i=2:m
        y_true(i)=y_true(i)+y_true(i-1);
    end
end
%% Kernel
CON={@covNoise};
CONhyp=log(0.1);
SE={@covSEiso};
SEhyp=[0;0];
SE2={@covSEard};
SE2hyp=[0;0;0];
RQ={@covRQiso};
RQhyp=[0;0;0];
RQ2={@covRQard};
RQ2hyp=[0;0;0;0;0;0];
MA={@covMaterniso,3};
MAhyp=[0;0];
MA2={@covMaternard,3};
MA2hyp=[0;0;0];
PER={@covPeriodic};
PERhyp=log([0.9;2;2]);
KER1={@covSum,{SE,CON}};KER1hyp=[SEhyp;CONhyp];
KER2={@covSum,{RQ,CON}};KER2hyp=[RQhyp;CONhyp];
KER0={@covProd,{SE2,RQ2}};KER0hyp=[SE2hyp;RQ2hyp];
KERp={@covSum,{KER0,MA2,CON}};KERphyp=[KER0hyp;MA2hyp;CONhyp];
KER={@covSum,{RQ2,CON}};KERhyp=[RQ2hyp;CONhyp];
%% GPR
meanfunc=[];                    % empty: don't use a mean function
covfunc=CON;              % Squared Exponental covariance function
likfunc=@likGauss;              % Gaussian likelihood
if(treatment==3)
    hyp=struct('mean',[],'cov',[5,7],'lik',-1);
else
    hyp=struct('mean',[],'cov',CONhyp,'lik',-1);
end
hyp=minimize(hyp,@gp,-100,@infGaussLik,meanfunc,covfunc,likfunc,X,y);
[y_m,y_sigma]=gp(hyp,@infGaussLik,meanfunc,covfunc,likfunc,X,y,X_star);
for i=1:m
    if(i==1)
        X_gpr=X;
        y_gpr=y;
    else
        X_gpr=[X;X_star(1:i-1,:)];
        y_gpr=[y;y_true(1:i-1,:)];
    end
    hyp=minimize(hyp,@gp,-100,@infGaussLik,meanfunc,covfunc,likfunc,X_gpr,y_gpr);
    [y_m(i),y_sigma(i)]=gp(hyp,@infGaussLik,meanfunc,covfunc,likfunc,X_gpr,y_gpr,X_star(i,:));
end
y_sigma=sqrt(y_sigma);
%% post-treatment
if(treatment==1)
    y_m=y_m+y_avg;
    y=y+y_avg;
    y_true=y_true+y_avg;
elseif(treatment==2)
    y=[y_0;y];
    for i=2:n
        y(i)=y(i)+y(i-1);
    end
    y_true(1)=y_true(1)+y(end);
    y_m(1)=y_m(1)+y(end);
    for i=2:length(y_m)
        y_m(i)=y_m(i)+y_true(i-1);
        y_true(i)=y_true(i)+y_true(i-1);
    end 
elseif(treatment==3)
    y_m=y_m-[y(n);y_m(1:end-1)];
    y_true=y_true-[y(n);y_true(1:end-1)];
    y=y-[0;y(1:end-1)];
end
%% calc evaluation indicator
MSE=sum((y_m-y_true).^2)/m;
sigma_avg=sum(y_sigma)/m;
C_avg=2*sum(qfunc(abs(y_m-y_true)./y_sigma))/m;
y_last=[y(end);y_true(1:end-1)];
T=(y_true-y_last).*(y_m-y_last);
T_avg=sum(T>0)/m; 
%% plot
factor_sigma=qfuncinv((1-confidence_interval)/2);
figure;
hold on;
fill([X_star(:,1);flipud(X_star(:,1))],[y_m-factor_sigma*y_sigma;flipud(y_m+factor_sigma*y_sigma)],[248,195,205]/255);
plot([X(end-5:end,1);X_star(:,1)],[y(end-5:end);y_true],'-*');
plot(X_star(:,1),y_true,'-+');
plot(X_star(:,1),y_m,'-ok','LineWidth',2);
title(['MSE=',num2str(MSE),'     \sigma=',num2str(sigma_avg),'     C=',num2str(C_avg),'     T=',num2str(T_avg)]);
hold off;