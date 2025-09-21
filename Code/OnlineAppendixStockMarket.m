function [ ThetDem, sigMdem, lamM, sigM, PDt, fCsig, rhoLE, fCrhoLE, XEalfa, TVol ] = OnlineAppendixStockMarket( nu, rhoA, rhoB, ft, sigY, Del )
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %       JF Round 1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stock Market return
    %
    if rhoA > rhoB
        error('check inputs');
    end    
    if (nu+rhoA) <= 0
        error('check inputs');
    end
    %
    phiA = 1/(rhoA+nu); phiB = 1/(rhoB+nu);
    PDt = ft*phiA + (1-ft)*phiB;
    %betAt = (rhoA+nu)*PDt;
    %betBt = (rhoB+nu)*PDt;
    %rt = ft*rhoA + (1-ft)*rhoB + muY - sigY^2 + nu*(1-at*betAt-(1-at)*betBt);
    ThetDem = Del*(0.5-ft);       % Market price of risk for demand shocks       
    sigMdem = Del*(phiA-phiB)*ft.*(1-ft)./PDt;
    lamM = sigY^2+sigMdem.*ThetDem;
    sigM = sqrt(sigMdem.^2+sigY^2);
    fCsig = (1+sqrt(phiA/phiB))^(-1);
    %
    fCrhoLE = 1/(1+sqrt(phiA/phiB));
    %
    if ft < fCrhoLE
        rhoLE = sigMdem/sigM;
    elseif ft > fCrhoLE
        rhoLE = -sigMdem/sigM;
        %rhoLE = (sigMdem/sigM)*(phiB*(1-ft).^2-phiA*ft.^2)/(phiA*ft.^2-phiB*(1-ft).^2);
    else
        rhoLE = 0;
    end
    %
    XEalfa = 2*phiB/(phiA-phiB)*sigMdem;
    TVol = abs(2*Del^2*ft.*(1-ft)*(phiB./(PDt.^2)).*(phiB*(1-ft).^2-phiA*ft.^2));
    %rhoLE = (sigMdem/sigM)*(phiB*(1-ft).^2-phiA*ft.^2)/abs(phiB*(1-ft).^2-phiA*ft.^2);
    %
end

