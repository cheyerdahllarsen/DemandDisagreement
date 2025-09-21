%clear all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%
VbarVEC = 0:1:10;
VbarVEC = [VbarVEC 12 14 16 18 20 25 30 40 50 60 70 80 90 100];
Mpaths = length(VbarVEC);
fSTAT = zeros(13,Mpaths);
Del_aSTAT = zeros(13,Mpaths);
DEL_bSTAT = zeros(13,Mpaths);
DEL_STAT =zeros(13,Mpaths);

PD_STAT = zeros(13,Mpaths);
theta_STAT = zeros(13,Mpaths);
r_STAT = zeros(13,Mpaths);
sigMt_STAT = zeros(13,Mpaths);
rpt_STAT = zeros(13,Mpaths);
stdRM_STAT = zeros(13,Mpaths);
RP_a_STAT = zeros(13,Mpaths);
RP_b_STAT = zeros(13,Mpaths);
corrRP_a = zeros(1,Mpaths);
corrRP_b = zeros(1,Mpaths);
corrPD_stdR = zeros(1,Mpaths);
corr_dPD_dstdR = zeros(1,Mpaths);
corr_PD_RP = zeros(1,Mpaths);



T = 1000;
dt = 1/12; %monthly
NT = T/dt;
COV_A  = zeros(Mpaths,NT+1); %Covariance of fst DEL
COV_B  = zeros(Mpaths,NT+1);
M_Ad = zeros(Mpaths,NT+1);
M_Bd = zeros(Mpaths,NT+1);
M_Af = zeros(Mpaths,NT+1);
M_Bf = zeros(Mpaths,NT+1);

DoSameShocks = 1;
DoPrelearning = 1;
DoCohort = 0;

T2 = 100000;%T+10000; %at least T+1
NT2 = T2/dt;

if DoSameShocks == 1
   dZtb = sqrt(dt)*randn(NT,1);
   dZt = sqrt(dt)*randn(NT2,1);
end


tic
for k=1:Mpaths
muY = 0.02;
sigY = 0.02;
rho_a = 0.001;
rho_b = 0.05;
kap_l = 0.01;
sig_l = 0.1;
sig_A = sig_l/kap_l;%the adjusted process for the learning
nu = 0.02;
rhoA = rho_a+nu;
rhoB = rho_b+nu;
% Vbar = 0.02;%sig_l^2./(2*kap_l);
% d = 8;
Vbar = VbarVEC(k);%sig_l^2/(2*kap_l);
d = 4;


lbar = 0;
lbar_a = d;
lbar_b = -d;
l0 = lbar;
if sig_l ==0
    DELa0 = 0;
    DELb0 = 0;
else
    DELa0 = kap_l*((lbar_a-lbar)/sig_l);
    DELb0 = kap_l*((lbar_b-lbar)/sig_l);
end
DELa = DELa0;
DELb = DELb0;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% BUILDING UP COHORTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


reduction = exp(-nu*dt);

alp0 = 0.5; %IMPORTANT TO SET IT TO THIS TO BE CONSISTENT
f0 = (rho_a+nu)/(rho_a+nu + rho_b+nu);
X0 = 1;
Xa0 = f0*X0;
Xb0 = (1-f0)*X0;
Ia = Xa0;
Ib = Xb0;

if DoSameShocks == 0
    dZtb = sqrt(dt)*randn(NT,1);
end
l_t = l0;
alp_t = alp0;
tau = dt;
XaVEC = zeros(NT,1);
XbVEC = zeros(NT,1);
% XaVEC2 = zeros(NT,1);
% XbVEC2 = zeros(NT,1);
fVECb = zeros(NT,1);
DELbaraVECb = zeros(NT,1);
DELbarbVECb = zeros(NT,1);
alpVECb = zeros(NT,1);
JUMPUP = 1000000000;
tic

for i=1:NT
    if sum(Ia)<0.000000000001
        Ia = Ia*JUMPUP;
        Ib = Ib*JUMPUP;
        disp('jump');
    end
    PartA = Ia.*exp(-(rho_a + 0.5*DELa.^2)*dt + DELa*dZtb(i));
    PartB = Ib.*exp(-(rho_b + 0.5*DELb.^2)*dt + DELb*dZtb(i));
    XaVEC(i) = sum(PartA);
    XbVEC(i) = sum(PartB);
    Xt = XaVEC(i)+XbVEC(i);
    DELbaraVECb(i) = sum(PartA.*DELa)/XaVEC(i);
    DELbarbVECb(i) = sum(PartB.*DELb)/XbVEC(i);


    fVECb(i) = XaVEC(i)/Xt;
    Phi_t = fVECb(i)/rhoA+ (1-fVECb(i))/rhoB;
    bet_a_t = rhoA*Phi_t;
    bet_b_t = rhoB*Phi_t;
    
    %simulate the demand shock%
    l_t = l_t + kap_l*(lbar-l_t)*dt + sig_l*dZtb(i);
    alpVECb(i) = 1/(1+exp(-l_t));
    if Vbar==0
        dDELa = 0;
        dDELb = 0;
    else
        dDELa = (PostVar(sig_A,Vbar,tau)/sig_A^2).*(-DELa*dt+ones(size(DELa))*dZtb(i));
        dDELb = (PostVar(sig_A,Vbar,tau)/sig_A^2).*(-DELb*dt+ones(size(DELb))*dZtb(i));
    end
    DELa = [DELa+dDELa DELa0];
    DELb = [DELb+dDELb DELb0];
    %end simulate
    newA = (1-reduction)*alp_t*bet_a_t*Xt;
    newB = (1-reduction)*(1-alp_t)*bet_b_t*Xt;
    
    Ia = [reduction*PartA newA];
    Ib = [reduction*PartB newB];
    alp_t = alpVECb(i);
    tau=[tau+dt 0];

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  END OF BUILDING UP COHORTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if DoPrelearning == 1
    PreT = 5;
    NpreT = PreT/dt;
    ww1 = sig_A^2/(sig_A^2+Vbar*PreT);
    ww2 = Vbar/(sig_A^2+Vbar*PreT);
    DELa00 = DELa0;
    DELb00 = DELb0;
    sumZ = sum(dZtb(end-NpreT+1));
    tau = tau+PreT;
end



Vt = (PostVar(sig_A,Vbar,tau(1:end-1))/sig_A^2);
Iv = ones(size(tau(1:end-1)));

if DoSameShocks == 0
   dZt = sqrt(dt)*randn(NT2,1);
end
l_t = l0;
alp_t = alpVECb(end);
XaVEC = zeros(NT2,1);
XbVEC = zeros(NT2,1);
XaVEC2 = zeros(NT2,1);
XbVEC2 = zeros(NT2,1);
fVEC = zeros(NT2,1);
DELbaraVEC = zeros(NT2,1);
DELbarbVEC = zeros(NT2,1);
alpVEC = zeros(NT2,1);

if DoCohort == 1
    cov_EXY_a = 0;
    cov_EXY_b = 0;
    cov_EY_a = 0;
    cov_EY_b = 0;
    cov_EX_a = 0;
    cov_EX_b = 0;
end

for i=1:NT2
    if Xt<0.000000000001
        Ia = Ia*JUMPUP;
        Ib = Ib*JUMPUP;
        disp('jump');
    end
    paa = -(rho_a + 0.5*DELa.^2)*dt + DELa*dZt(i);
    pbb = -(rho_b + 0.5*DELb.^2)*dt + DELb*dZt(i);
    PartA = Ia.*exp(paa);
    PartB = Ib.*exp(pbb);
    XaVEC(i) = sum(PartA);
    XbVEC(i) = sum(PartB);
    Xt = XaVEC(i)+XbVEC(i);

    DELbaraVEC(i) = PartA*DELa'/XaVEC(i);
    DELbarbVEC(i) = PartB*DELb'/XbVEC(i);
    
    if DoCohort == 1
        cov_EXY_a = cov_EXY_a + PartA.*DELa/XaVEC(i);
        cov_EXY_b = cov_EXY_b + PartB.*DELb/XbVEC(i);
        cov_EX_a =  cov_EX_a+PartA/XaVEC(i);
        cov_EX_b =  cov_EX_b+PartB/XbVEC(i);
        cov_EY_a =  cov_EY_a+DELa;
        cov_EY_b =  cov_EY_b+DELb;
    end

    fVEC(i) = XaVEC(i)/Xt;
    Phi_t = fVEC(i)/(rho_a+nu)+ (1-fVEC(i))/(rho_b+nu);
    bet_a_t = (rho_a+nu)*Phi_t;
    bet_b_t = (rho_b+nu)*Phi_t;

    %simulate the demand shock%
    l_t = l_t + kap_l*(lbar-l_t)*dt + sig_l*dZt(i);
    alpVEC(i) = 1/(1+exp(-l_t));
    if Vbar==0
        dDELa = 0;
        dDELb = 0;
    else
        dDELa = Vt.*(-DELa(2:end)*dt+Iv*dZt(i)); %SPEED UP BY PRECALC!
        dDELb = Vt.*(-DELb(2:end)*dt+Iv*dZt(i));
    end
    
    if DoPrelearning == 1
        DELa0 = ww1*DELa00 + ww2*sumZ;
        DELb0 = ww1*DELb00 + ww2*sumZ;
        if i<NpreT
            sumZ = sumZ-dZtb(end-NpreT+1+i)+dZt(i);
        else
            sumZ = sumZ-dZt(i-NpreT+1)+dZt(i);
        end
    end
    
    DELa = [DELa(2:end)+dDELa DELa0];
    DELb = [DELb(2:end)+dDELb DELb0];
    %end simulate
    newA = (1-reduction)*alp_t*bet_a_t*Xt;
    newB = (1-reduction)*(1-alp_t)*bet_b_t*Xt;
    
    Ia = [reduction*PartA(2:end) newA];
    Ib = [reduction*PartB(2:end) newB];
    
    alp_t = alpVEC(i);
   
end
toc
%Calculating oustide 
PDt = fVEC/rhoA+(1-fVEC)/rhoB;
theta_alp = -(DELbaraVEC.*fVEC+DELbarbVEC.*(1-fVEC));
rt = fVEC*rho_a+(1-fVEC)*rho_b+muY-sigY^2 +nu*(1-rhoA*alpVEC.*PDt-rhoB*(1-alpVEC).*PDt);
DEL = DELbaraVEC-DELbarbVEC;
sigMt = (1/rhoA-1/rhoB)*(1./PDt).*fVEC.*(1-fVEC).*DEL;
rpt = sigY^2 + sigMt.*theta_alp;
stdRm = (sigY^2+sigMt.^2).^0.5;

RP_a = rpt+sigMt.*DELbaraVEC;
RP_b = rpt+sigMt.*DELbarbVEC;
corrRP_a(k) = corr(RP_a,rpt);
corrRP_b(k) = corr(RP_b,rpt);
corrPD_stdR(k) = corr(stdRm,PDt);
corr_dPD_dstdR(k) = corr(diff(stdRm),diff(PDt));
corr_PD_RP(k) = corr(PDt,rpt);


if DoCohort==1  
    M_Ad(k,:) = (1/NT2)*cov_EY_a;  
    M_Bd(k,:) = (1/NT2)*cov_EY_b;
    M_Af(k,:) = (1/NT2)*cov_EX_a;
    M_Bf(k,:) = (1/NT2)*cov_EX_b;
    COV_A(k,:) = (1/NT2)*cov_EXY_a-M_Ad(k,:).*M_Af(k,:);
    COV_B(k,:) = (1/NT2)*cov_EXY_b-M_Bd(k,:).*M_Bf(k,:);
end


%some more



RP_a_STAT(:,k) = sumstatforDD(RP_a);
RP_b_STAT(:,k) = sumstatforDD(RP_b);
fSTAT(:,k) = sumstatforDD(fVEC);
Del_aSTAT(:,k) = sumstatforDD(DELbaraVEC);
DEL_bSTAT(:,k) = sumstatforDD(DELbarbVEC);
DEL_STAT(:,k) =sumstatforDD(DELbaraVEC-DELbarbVEC);
PD_STAT = sumstatforDD(PDt);
theta_STAT(:,k) =sumstatforDD(theta_alp);
r_STAT(:,k) = sumstatforDD(rt);
sigMt_STAT(:,k) = sumstatforDD(sigMt);
rpt_STAT(:,k) = sumstatforDD(rpt);
stdRM_STAT(:,k) = sumstatforDD(stdRm);
k
[sumstatforDD(fVEC) sumstatforDD(DELbaraVEC) sumstatforDD(DELbarbVEC) sumstatforDD(DELbaraVEC-DELbarbVEC)]
end
save('TEST_Run_ForConferencesNOV18_d4_N1.mat');

[mean(fSTAT,2) mean(Del_aSTAT,2) mean(DEL_bSTAT,2) mean(DEL_STAT,2)]
