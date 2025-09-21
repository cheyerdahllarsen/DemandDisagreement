function [ fClamMax, fClamMaxVal, fClamMin, fClamMinVal ] = OnlineAppendixStockMarketCriticalValues( nu, rhoA, rhoB, sigY, Del )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    if rhoA > rhoB
        error('check inputs');
    end    
    if (nu+rhoA) <= 0
        error('check inputs');
    end
    %
    [ fClamMax, fClamMaxVal ] = fminbnd(@myfun1, 0, 0.5);
    [ fClamMin, fClamMinVal ] = fminbnd(@myfun2, 0.5, 1);
    %
    fClamMaxVal = -fClamMaxVal;
    %
    function RESmax = myfun1(x)
        [ ~, ~, RESmax ] = OnlineAppendixStockMarket( nu, rhoA, rhoB, x, sigY, Del );
        RESmax = -RESmax;
    end
    %
    function RESmin = myfun2(x)
        [ ~, ~, RESmin ] = OnlineAppendixStockMarket( nu, rhoA, rhoB, x, sigY, Del );
    end
    %
end

