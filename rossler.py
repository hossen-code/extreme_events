function dy=rossler(t,y)
t
N=round(length(y)/3);
dy=zeros(3*N,1);

%parameters
% a=.2; b=.2; c=5.7;
a=.38; b=.3; c=4.82;
 
x1=y(1:N,1);
x2=y(N+1:2*N,1);
x3=y(2*N+1:3*N,1);

dy(1:N,1)      = -x2-x3;
dy(N+1:2*N,1)  = x1+a*x2;
dy(2*N+1:3*N,1)= b+x3.*(x1-c);
