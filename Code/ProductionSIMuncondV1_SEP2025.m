clear all;
costVEC = 0:1:1000;
costVEC(1) = 0.01;
Ncost = length(costVEC);
dt = 1/12;
T = 10000;
NT = T/dt;
epsout = 0.0000001;
dzk = sqrt(dt)*randn(NT,1);
dza = sqrt(dt)*randn(NT,1);

DataAP = zeros(Ncost,5);

corr1 = zeros(Ncost,2);
corr5 = zeros(Ncost,3);
corr10 = zeros(Ncost,3);

for j=1:Ncost
    
rho = [0.001 0.05];
adjCost = costVEC(j); %capital adjustment
abar = 0.05; %TFP
del = -0.02; %here we should haved a mu term here
sigK = 0.033;

%Simulate the f (they don't depend on the production side here)
nu = 0.02;  %birth and death intensity
rhoA = rho(1) + nu;
rhoB = rho(2) + nu;
a = rhoB/rhoA;
Drho = rho(2)-rho(1);
kap = 0.01; 
sigl = 0.1;
d = 4;
lbar_e = 0; %econometrician
lbar_a = lbar_e+d;  
lbar_b = lbar_e-d;
if sigl>0
    DELa = (kap/sigl)*(lbar_a-lbar_e);
    DELb = (kap/sigl)*(lbar_b-lbar_e);
    DEL = DELa-DELb;
else
    DELa = 0;
    DELb = 0;
    DEL  = 0;
end
Lt = zeros(NT,1); %the process for alpha
alpt = zeros(NT,1);
ft = zeros(NT,1);
Kt = zeros(NT,1);
it = zeros(NT,1);
Ct = zeros(NT,1);
Phit = zeros(NT,1);
Qt = zeros(NT,1);
PDt = zeros(NT,1);
f0 = 1/(1+exp(-lbar_e)); %should maybe choose the acutal dis for alpa
l0 = lbar_e;
K0 =0.01;

for i=1:NT
    if i==1
        alp  = 1/(1+exp(-lbar_e));
        betA = rhoA*(f0/rhoA+(1-f0)/rhoB);
        betB = rhoB*(f0/rhoA+(1-f0)/rhoB);
        DELbar = f0*DELa+(1-f0)*DELb;
        muf = nu*(alp*betA*(1-f0)-(1-alp)*betB*f0)+Drho*f0*(1-f0)+f0*(1-f0)*DELbar*(-DEL);
        sigf = f0*(DELa-DELbar);
        df = muf*dt + sigf*dza(i);
        ft(i) = f0 +df;
        Lt(i) = l0 + kap*(lbar_e-l0)*dt + sigl*dza(i);
        
        Phi = f0/(rho(1)+nu) + (1-f0)/(rho(2)+nu);
        Q = (1+adjCost*abar)./(1+adjCost*Phi.^(-1));
        lt = (Q-1)/adjCost;
        phiovl = (1/adjCost)*log(1+adjCost*lt);
        muK = phiovl-del;
        dK = muK*K0*dt + sigK*K0*dzk(i);
        Kt(i)  = K0 + dK;
        alpt(i) = 1/(1+exp(-Lt(i)));
        Phit(i) = ft(i)/(rho(1)+nu) + (1-ft(i))/(rho(2)+nu);
        Qt(i) = (1+adjCost*abar)./(1+adjCost*Phit(i).^(-1));
        it(i) = (Qt(i)-1)/adjCost;
        Ct(i) = (abar-it(i))*Kt(i);
        PDt(i) = Qt(i)/abar;
     else
        alp  = 1/(1+exp(-Lt(i-1)));
        betA = rhoA*(ft(i-1)/rhoA+(1-ft(i-1))/rhoB);
        betB = rhoB*(ft(i-1)/rhoA+(1-ft(i-1))/rhoB);
        DELbar = ft(i-1)*DELa+(1-ft(i-1))*DELb;
        muf = nu*(alpt(i-1)*betA*(1-ft(i-1))-(1-alpt(i-1))*betB*ft(i-1))+Drho*ft(i-1)*(1-ft(i-1))+ft(i-1)*(1-ft(i-1))*DELbar*(-DEL);
        sigf = ft(i-1)*(DELa-DELbar);
        df = muf*dt + sigf*dza(i);
        ft(i) = ft(i-1) +df;
         if ft(i)<0
            ft(i) = epsout;
        end
        if ft(i)>1
            ft(i) = 1-epsout;
        end
        Lt(i) = Lt(i-1) + kap*(lbar_e-Lt(i-1))*dt + sigl*dza(i);
        alpt(i) = 1/(1+exp(-Lt(i)));
        
        
        phiovl = (1/adjCost)*log(1+adjCost*lt);
        muK = phiovl-del;
        dK = muK*Kt(i-1)*dt + sigK*Kt(i-1)*dzk(i);
        Kt(i)  = Kt(i-1) + dK;
        alpt(i) = 1/(1+exp(-Lt(i)));
        Phit(i) = ft(i)/(rho(1)+nu) + (1-ft(i))/(rho(2)+nu);
        Qt(i) = (1+adjCost*abar)./(1+adjCost*Phit(i).^(-1));
        it(i) = (Qt(i)-1)/adjCost;
        Ct(i) = (abar-it(i))*Kt(i);
        PDt(i) = Qt(i)/abar;
    end
end


%End simulating 

fVEC = ft;
EfDEL = (fVEC*DELa+(1-fVEC)*DELb);
Theta_alp_E = -EfDEL;
Phi = fVEC/(rho(1)+nu) + (1-fVEC)/(rho(2)+nu);
sig_Phi = ((1/(rho(1)+nu) - 1/(rho(2)+nu))./Phi).*fVEC.*(1-fVEC).*DEL;

RP_alp_E = sig_Phi.*Theta_alp_E;
sigCalp = -P
hi./(adjCost+Phi).*sig_Phi;
Theta_alp_p = Theta_alp_E+sigCalp;
sig_M_alp_p = sig_Phi + sigCalp;
RP_alp_P = sig_M_alp_p.*Theta_alp_p;




Qt = (1+adjCost*abar)./(1+adjCost*Phi.^(-1));
lt = (Qt-1)/adjCost;
CoverY = (abar-lt)/abar;
Ct = (abar-lt).*Kt;
phiovl = (1/adjCost)*log(1+adjCost*lt);
muK = phiovl-del;
j

DataAP(j,1) = mean(RP_alp_P + sigK^2); %check
DataAP(j,2) = mean(sqrt(sig_M_alp_p.^2+sigK^2)); %check
DataAP(j,3) = mean(sig_M_alp_p);
DataAP(j,4) = mean(sqrt(sigCalp.^2 +sigK^2));
DataAP(j,5) = mean(sigCalp);





%log consumption growth
dlogC = log(Ct(2:end)./Ct(1:end-1));
dlogY = log(Kt(2:end)./Kt(1:end-1));
Rt = (Qt(2:end).*Kt(2:end)+abar*Kt(2:end)*dt)./(Qt(1:end-1).*Kt(1:end-1));
rt = log(Rt);


NP = 12;
Ny = floor((NT-1)/NP);
DAT1 = zeros(Ny,3);
counter = 1;
for i=1:Ny
    DAT1(i,1) = sum(rt(1+counter:counter+NP-1));
    DAT1(i,2) = sum(dlogC(1+counter:counter+NP-1));
    DAT1(i,3) =  sum(dlogY(1+counter:counter+NP-1));
    counter = counter+NP;
end

corr1(j,1) = corr(DAT1(:,1),DAT1(:,2));
corr1(j,2) = corr(DAT1(:,1),DAT1(:,3));


DAT2 = zeros(Ny-5,3);

Ny2 = Ny-5;
counter = 1;
for i=1:Ny2
    DAT2(i,1) = sum(DAT1(counter:counter+4,1));
    DAT2(i,2) = sum(DAT1(counter:counter+4,2));
    DAT2(i,3) = sum(DAT1(counter:counter+4,3));

    counter = counter+1;
end

corr5(j,1) =corr(DAT2(:,1),DAT2(:,2));
corr5(j,2) =corr(DAT2(:,1),DAT2(:,3));


DAT3 = zeros(Ny-10,3);

Ny2 = Ny-10;
counter = 1;
for i=1:Ny2
    DAT3(i,1) = sum(DAT1(counter:counter+9,1));
    DAT3(i,2) = sum(DAT1(counter:counter+9,2));
    DAT3(i,3) = sum(DAT1(counter:counter+9,3));

    counter = counter+1;
end

corr10(j,1) =corr(DAT3(:,1),DAT3(:,2));
corr10(j,2) =corr(DAT3(:,1),DAT3(:,3));


end
figure;
plot(costVEC,corr1(:,1),costVEC,corr5(:,1),costVEC,corr10(:,1));
legend('1 year','5 year','10 year');

save('ProdUnconditionalDel08V2_TEST.mat');


