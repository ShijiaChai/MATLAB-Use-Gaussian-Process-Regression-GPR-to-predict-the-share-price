function [f,grd]=Kernel_PER(X_r,y,sigma2_N,hyper_parameter)
hyper_parameter=exp(hyper_parameter);
A_PER=hyper_parameter(1);
F_PER=hyper_parameter(2);
n=length(y);
K=A_PER*exp((sin(F_PER*X_r)).^2)+sigma2_N*eye(n);
if(cond(K)>1e4)
    display('!!');
end
K_inv=inv(K);
alpha=K_inv*y;
f=y.'*alpha+log(abs(det(K)));
if(nargout>1)
    grd=zeros(length(hyper_parameter),1);
    K_grd=alpha*alpha.'-K_inv;
    A_PER_d=exp((sin(F_PER*X_r)).^2);
    F_PER_d=A_PER*F_PER*sin(2*F_PER*X_r).*exp((sin(F_PER*X_r)).^2);
    grd(1)=A_PER*trace(K_grd*A_PER_d);
    grd(2)=F_PER*trace(K_grd*F_PER_d);
end
end