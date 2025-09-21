function [ fT, alfaT, Xt, lT ] = OnlineAppendixSimulateConsumptionShareDogmaticLongRunMean(kap, lbar, sigl, d, nu, rho, drho, dZ, dt, alfa0, f0)

    % simulate consumption share
    % dogmatic long run mean
    %
    rhoA = rho - drho/2;
    rhoB = rho + drho/2;
    phiA = 1/(rhoA+nu);
    phiB = 1/(rhoB+nu);
    %a = rhoB/rhoA;
    lbar_a = lbar+d;  
    lbar_b = lbar-d;

    %Estimation errors
    if sigl>0
        DELa = (kap/sigl)*(lbar_a-lbar);
        DELb = (kap/sigl)*(lbar_b-lbar);
        DEL = DELa-DELb;
    else
        DELa = 0;
        DELb = 0;
        DEL = DELa-DELb;
    end
    NT = length(dZ);
    %
    l0 = log(alfa0/(1-alfa0));
    lT = -999*ones(NT,1);
    alfaT = -999*ones(NT,1);
    % ft = 1/(1+e^(-Xt))
    X0 = log(f0/(1-f0));
    Xt = -999*ones(NT,1);
    %
    sigX = DEL;
    fT = -999*ones(NT,1);
    %
    for i=1:NT
        if i==1
            lT(i) = l0 + kap*(lbar-l0)*dt + sigl*dZ(i);
            muX = nu*alfa0*(1+(phiB/phiA)*exp(-X0)) - nu*(1-alfa0)*(1+(phiA/phiB)*exp(X0)) + drho - 0.5*DEL*(DELa+DELb);
            Xt(i) = X0 + muX*dt + sigX*dZ(i);
            fT(i) = 1/(1+exp(-Xt(i)));
            alfaT(i) = 1/(1+exp(-lT(i)));
        else
            lT(i) = lT(i-1) + kap*(lbar-lT(i-1))*dt + sigl*dZ(i);
            muX = nu*alfaT(i-1)*(1+(phiB/phiA)*exp(-Xt(i-1))) - nu*(1-alfaT(i-1))*(1+(phiA/phiB)*exp(Xt(i-1))) + drho - 0.5*DEL*(DELa+DELb);
            Xt(i) = Xt(i-1)+ muX*dt + sigX*dZ(i);
            fT(i) = 1/(1+exp(-Xt(i)));
            alfaT(i) = 1/(1+exp(-lT(i)));
        end
    end
    %
 %


