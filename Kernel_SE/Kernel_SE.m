function [f,grd]=Kernel_SE(X_r,y,sigma2_N,hyper_parameter)
hyper_parameter=exp(hyper_parameter);
A_SE=hyper_parameter(1);
M_SE=hyper_parameter(2);
n=length(y);
K=A_SE*exp(-M_SE*X_r)+sigma2_N*eye(n);
if(cond(K)>1e4)
    display('!!');
end
K_inv=inv(K);
alpha=K_inv*y;
f=y.'*alpha+log(det(K));
if(nargout>1)
    grd=zeros(length(hyper_parameter),1);
    K_grd=alpha*alpha.'-K_inv;
    A_SE_d=exp(-M_SE*X_r);
    M_SE_d=-A_SE*X_r.*exp(-M_SE*X_r);
    grd(1)=A_SE*trace(K_grd*A_SE_d);
    grd(2)=M_SE*trace(K_grd*M_SE_d);
end
end