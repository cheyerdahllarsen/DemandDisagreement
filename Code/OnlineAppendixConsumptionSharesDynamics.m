function [ muf, sigf, OLG, PREF, DIS ] = OnlineAppendixConsumptionSharesDynamics(nu, rhoA, rhoB, Del, ft, at)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %       JF Round 1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % consumption share dynamics
    %
    phiA = 1/(nu+rhoA);
    phiB = 1/(nu+rhoB);
    phit = phiA*ft+phiB*(1-ft);
    betAt = phit/phiA;
    betBt = phit/phiB;
    sigf = ft.*(1-ft)*Del;
    OLG = nu*(at.*betAt.*(1-ft)-(1-at).*betBt.*ft);
    PREF = (rhoB-rhoA)*ft.*(1-ft);
    DIS = Del^2*(0.5-ft).*ft.*(1-ft);
    muf = OLG+PREF+DIS;
end

