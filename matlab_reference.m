clear all
clc
close all

format long
% nonlinear quadrotor model

Jxx=6.86*10^-5;  Jyy=9.2*10^-5;  Jzz=1.366*10^-4;
m=0.068; kt=0.01;  kq=7.8263*0.0001; b=0.062/sqrt(2);
g=9.81;

t1=(Jyy-Jzz)/Jxx;
t2=(Jzz-Jxx)/Jyy;
t3=(Jxx-Jyy)/Jzz;

 dt=0.001;
 tend=5;
% initial conditions
x=0;y=0;z=0;xdot=0;ydot=0;zdot=0;u=0;v=0;w=0;
p=0;q=0;r=0;phi=0;theta=0;psi=0;
sump=0;sumt=0;sumpsi=0; sumz=0;
fflag=0;
% max motor rpm
Tmax=2.0*m*g;
nmax=sqrt(Tmax/(4*kt));
txymax=(Tmax/4)*2*b;
tzmax=2*kq*nmax*nmax;  % max. yawing torque
%tfault=500000; % fault starting count
th=0.0000001;
for i=1:tend/dt
% rotational motion

% controller for attitude tracking
% outer loop PI (k1 and ki) and inner loop P control (k2)

% phi loop

 k2=0.1; k1=1.0;  ki=0.4*0.01;

a1=k2/Jxx;   b1=a1*k1;  c1=a1*ki;

% closed loop  poles
phir=10*pi/180;
%damp([1,a1,b1,c1])
sump=sump+(phir-phi);
pr=k1*(phir-phi)+ki*sump*dt;
% if(i>tfault)
%     pr=0;
% end

tx=k2*(pr-p);

if(tx>txymax)
    tx=txymax;
end

if(tx<-txymax)
    tx=-txymax;
end
if(abs(tx)<th)
    tx=0;
end

%---------------------------

% theta loop

 k21=0.1; k11=1.0;  ki1=0.4*0.01;



% closed loop  poles

%damp([1,a1,b1,c1])
thetar=-5*pi/180;

sumt=sumt+(thetar-theta);
qr=k11*(thetar-theta)+ki1*sumt*dt; % q ref


ty=k21*(qr-q);

if(ty>txymax)
    ty=txymax;
end

if(ty<-txymax)
    ty=-txymax;
end

if(abs(ty)<th)
    ty=0;
end
%---------------------------
% psi loop

 k22=0.1; k12=1.0;  ki2=0.4*0.01;



% closed loop  poles

%damp([1,a1,b1,c1])
psir=5*pi/180;
sumpsi=sumpsi+(psir-psi);
rref=k12*(psir-psi)+ki2*sumpsi*dt;  % r ref - CORRECTED: using sumpsi instead of sumt

tz=k22*(rref-r);

if(tz>tzmax)
    tz=tzmax;
end

if(tz<-tzmax)
    tz=-tzmax;
end

if(abs(tz)<th)
    tz=0;
end

%---------------------------
% altitude hold
zr=-5.0;
kv=1.0; kz1=2.0; kz2=0.1*1.5;  % FIXED: Changed from -1.0 to +1.0 for proper control
sumz=sumz+(zr-z);
vzr=kz1*(zr-z)+kz2*sumz*dt;
T=kv*(vzr-zdot);
fx=0; fy=0;
if(T<0.1*m*g)
    T=0.1*m*g;
end

if(T>Tmax)
    T=Tmax;
end



%----------------------------------
% thrust-torque conversion

B=[tx;ty;tz;T];



A=[b*kt,-b*kt,-b*kt,b*kt;
    b*kt,b*kt,-b*kt,-b*kt;
    kq,-kq,kq,-kq;
    kt,kt,kt,kt];


C=inv(A)*B;



n1=sqrt(C(1));n2=sqrt(C(2));n3=sqrt(C(3));n4=sqrt(C(4));



%---- Dynamics-----------

pdot=t1*q*r+tx/Jxx-2*p;

qdot=t2*p*r+ty/Jyy-2*q;

rdot=t3*p*q+tz/Jzz-2*r;

p=p+pdot*dt; q=q+qdot*dt; r=r+rdot*dt;

phidot=p+sin(phi)*tan(theta)*q+cos(phi)*tan(theta)*r;
thetadot=cos(phi)*q-sin(phi)*r;
psidot=sin(phi)*q/cos(theta)+cos(phi)*r/cos(theta);

phi=phi+phidot*dt;
theta=theta+thetadot*dt;
psi=psi+psidot*dt;

if(phi>1*pi)
    phi=phi-2*pi;
end

if(phi<-1*pi)
    phi=phi+2*pi;
end

if(theta>1*pi)
    theta=theta-2*pi;
end

if(theta<-1*pi)
    theta=theta+2*pi;
end

if(psi>1*pi)
    psi=psi-2*pi;
end

if(psi<-1*pi)
    psi=psi+2*pi;
end






% Linear motion

fz=-T;

udot=r*v-q*w+fx/m- g*sin(theta)-0.1*u;

vdot=p*w-r*u+fy/m+ g*cos(theta)*sin(phi)-0.1*v;

wdot=q*u-p*v+fz/m+ g*cos(theta)*cos(phi)-0.1*w;

u=u+udot*dt;
v=v+vdot*dt;
w=w+wdot*dt;



xdot = ((cos(psi)*cos(theta))*u+ (cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi))*v+ (sin(psi)*sin(phi)+cos(psi)*sin(theta)*cos(phi))*w);
ydot = ((sin(psi)*cos(theta))*u + (cos(psi)*cos(phi) + sin(psi)*sin(theta)*sin(phi))*v + (sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi))*w);
zdot = -1*(sin(theta)*u - cos(theta)*sin(phi)*v - cos(theta)*cos(phi)*w);

x=x+xdot*dt;
y=y+ydot*dt;
z=z+zdot*dt;


if(z>0)
    break;
end




tim(i)=i*dt;
% storing
hstore(i)=-z;
phistore(i)=phi*180/pi;
thetastore(i)=theta*180/pi;
psistore(i)=psi*180/pi;
thetard(i)=thetar*180/pi;
phird(i)=phir*180/pi;
hdstore(i)=-zr;
psird(i)=psir*180/pi;


end
figure(1)
plot(tim,hstore);
grid on
hold on
plot(tim,hdstore,'r')
xlabel('time (s)')
ylabel('height (m)')

figure(2)
plot(tim,phistore);
grid on
hold on
plot(tim,phird,'r')
xlabel('time (s)')
ylabel('\phi (deg)')


figure(3)
plot(tim,thetastore);
grid on
hold on
plot(tim,thetard,'r')
xlabel('time (s)')
ylabel('\theta (deg)')


figure(4)
plot(tim,psistore);
grid on
hold on
plot(tim,psird,'r')
xlabel('time (s)')
ylabel('\psi (deg)')