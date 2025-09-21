% Note This file create figures in the online appendix for sections VIII - XII
% The file loads the paths from our simulations, but you can create your own paths by running the python code for the main paper. 
clear; close all; clc;
%
sl = 9;
%
% Derivations
%
% sl = 1 ......... The consumption share drift and volatility as a function of the consumption share ft
% sl = 2 .......   Stock Market Volatility graph as a function of the consumption share ft 
% sl = 3 .......   Stock Market Risk Premium as a function of the consumption share ft 
% sl = 4 .......   Trading volume as a function of the consumption share
% sl = 5 .......   Stock Market Return Predictability  as a function of the consumption share ft 
% sl = 6 .......   Leverage Effect for Stock Market Returns as a function of the consumption share ft 
% sl = 7 .......   Unconditional mean and volatilities as a function of disagreement
% sl = 8 .......   Predictability, correlation puzzle, and leverage effect
% sl = 9 .......   Unconditional distribution --- histograms, sl = 9.1
%
base_font_size = 18;
% 
%




    % -----------------------------------------------------------------------------------
if sl == 1          % The consumption share drift and volatility as a function of the consumption share ft
    % -----------------------------------------------------------------------------------
    nu = 0.02;
    kap = 0.01;
    lbar = 0; 
    sigl = 0.1;
    rhoA = 0;
    rhoB = 0.05;
    rho = mean([rhoA;rhoB]);
    drho = rhoB-rhoA;
    sigY = 0.033;
    muY = 0.02;
    at = 0.5;
    %
    dim = 1000;
    fmin = 0;
    fmax = 1;
    ftVec = linspace(fmin, fmax,dim);
    DelVec = [0; 0.4; 0.8];
    dim2 = length(DelVec);
    %
    for i = 1:dim
        ft = ftVec(i);
        for j = 1:dim2
            Del = DelVec(j);
           [ muf(i,j), sigf(i,j), OLG(i,j), PREF(i,j), DIS(i,j) ] = OnlineAppendixConsumptionSharesDynamics(nu, rhoA, rhoB, Del, ft, at);
        end
    end
    %
    
    
        %%
        figure;
        lines_str = {'-.r', '--b', 'k' };
        for j = 1:dim2
            plot(ftVec, sigf(:,j), lines_str{j}, 'LineWidth', 2);
            legendInfo{j} = ['$\Delta$ = ' num2str(DelVec(j)) ];
            hold on
        end
        legend( legendInfo, 'Location', 'Best', 'Interpreter','LaTex'); 
        xlabel('Consumption share, f_t', 'FontSize',base_font_size);
        ylabel('Consumption Share Vola', 'FontSize',base_font_size);
        grid on
        %
        %%
        figure;
        lines_str = {'-.r', '--b', 'k' };
        for j = 1:dim2
            plot(ftVec, OLG(:,j), lines_str{j}, 'LineWidth', 2);
            legendInfo{j} = ['$\Delta$ = ' num2str(DelVec(j)) ];
            hold on
        end
        legend( legendInfo, 'Location', 'Best', 'Interpreter','LaTex'); 
        xlabel('Consumption share, f_t', 'FontSize',base_font_size);
        ylabel('Drift of Consumption Share, OLG', 'FontSize',base_font_size);
        grid on
        %
        %%
        figure;
        lines_str = {'-.r', '--b', 'k' };
        for j = 1:dim2
            plot(ftVec, PREF(:,j), lines_str{j}, 'LineWidth', 2);
            legendInfo{j} = ['$\Delta$ = ' num2str(DelVec(j)) ];
            hold on
        end
        legend( legendInfo, 'Location', 'Best', 'Interpreter','LaTex'); 
        xlabel('Consumption share, f_t', 'FontSize',base_font_size);
        ylabel('Drift of Consumption Share PREF part', 'FontSize',base_font_size);
        grid on
        %
        %%
        figure;
        lines_str = {'-.r', '--b', 'k' };
        for j = 1:dim2
            plot(ftVec, DIS(:,j), lines_str{j}, 'LineWidth', 2);
            legendInfo{j} = ['$\Delta$ = ' num2str(DelVec(j)) ];
            hold on
        end
        legend( legendInfo, 'Location', 'Best', 'Interpreter','LaTex'); 
        xlabel('Consumption share, f_t', 'FontSize',base_font_size);
        ylabel('Drift of Consumption Share, DIS', 'FontSize',base_font_size);
        grid on
        %
        %%
        figure;
        lines_str = {'-.r', '--b', 'k' };
        for j = 1:dim2
            plot(ftVec, muf(:,j), lines_str{j}, 'LineWidth', 2);
            legendInfo{j} = ['$\Delta$ = ' num2str(DelVec(j)) ];
            hold on
        end
        legend( legendInfo, 'Location', 'Best', 'Interpreter','LaTex'); 
        xlabel('Consumption share, f_t', 'FontSize',base_font_size);
        ylabel('Drift of Consumption Share', 'FontSize',base_font_size);
        grid on
        %
        
        %%
        %
        ymin = -1; ymax = 2;
        %
        figure;
        ddim = 50;
        p1 = plot(ftVec, sigf(:,1), 'k', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on;
        p2 = plot(ftVec, sigf(:,2), '--bo', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p3 = plot(ftVec, sigf(:,3), '-.rx', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        %hold on
        %p4 = plot([fCcovLE(2), fCcovLE(2)], [ymin, 0], ':k', [fmin, fCcovLE(2)], [0, 0], ':k', 'LineWidth', 2);
        %hold on
        %p5 = plot([fCcovLE(3), fCcovLE(3)], [ymin, 0], ':k', [fmin, fCcovLE(3)], [0, 0], ':k', 'LineWidth', 2);
        hold off
        %
        xlabel('Consumption Share $f_t$', 'Interpreter','LaTex', 'FontSize',base_font_size);
        %ylabel('Return Volatility $\sigma_{S,t}^{\alpha}$ in %', 'Interpreter','LaTex', 'FontSize', base_font_size );
        %ylabel('Risk Premium (%)', 'Interpreter','LaTex', 'FontSize', base_font_size );
        ylabel('Volatility $\sigma_{f,t}$', 'Interpreter','LaTex', 'FontSize', base_font_size );
        title('Dynamics of the Consumption Share', 'FontSize', base_font_size, 'FontWeight', 'bold');             
        legend([ p1, p2, p3 ],  {'$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8'},  'Location', 'Northeast', 'Interpreter','LaTex', 'FontSize', 14) ; 
        ylim([-0.01; 0.225]);
        %legend([ p1, p2, p3 ],  {'$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8'},  'Location', 'Northeast', 'Interpreter','LaTex', 'FontSize', 14) ; 
        %axis( [ fmin fmax ymin  ymax ] );
        %set(gca,'XTick', [0; 0.1; 0.25; fCcovLE(2); 0.5; 0.65; 0.75; 0.9; 1]);            
        %set(gca,'XTickLabel',{'0', '0.1', '0.25', '0.35', '0.5', '0.65', '0.75', '0.9', '1'})        
        %set(gca,'YTick', [-0.007; fClamMinVal(3); -0.0025; fClamMinVal(2); 0; lamM(1,1); 0.0025; fClamMaxVal(2);  0.005;  0.0075; 0.01; 0.0125; fClamMaxVal(3);  0.015]);            
        %set(gca,'YTickLabel',{'-0.70', '-0.53', '-0.25', '-0.05', '0', '0.11', '0.25', '0.42', '0.50', '0.75', '1.00', '1.25', '1.35', '1.5'})        
        %set(gca,'YTickLabel',{'-70', '-53', '-25', '-5', '0', '11', '25', '42', '50', '75', '100', '125', '135', '150'})        
        grid on
        %%
        %
        figure;
        ddim = 50;
        p1 = plot(ftVec, muf(:,1), 'k', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on;
        p2 = plot(ftVec, muf(:,2), '--bo', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p3 = plot(ftVec, muf(:,3), '-.rx', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        %hold on
        %p4 = plot([fCcovLE(2), fCcovLE(2)], [ymin, 0], ':k', [fmin, fCcovLE(2)], [0, 0], ':k', 'LineWidth', 2);
        %hold on
        %p5 = plot([fCcovLE(3), fCcovLE(3)], [ymin, 0], ':k', [fmin, fCcovLE(3)], [0, 0], ':k', 'LineWidth', 2);
        hold off
        %
        xlabel('Consumption Share $f_t$', 'Interpreter','LaTex', 'FontSize',base_font_size);
        %ylabel('Return Volatility $\sigma_{S,t}^{\alpha}$ in %', 'Interpreter','LaTex', 'FontSize', base_font_size );
        %ylabel('Risk Premium (%)', 'Interpreter','LaTex', 'FontSize', base_font_size );
        ylabel('Drift $\mu_{f,t}$', 'Interpreter','LaTex', 'FontSize', base_font_size );
        title('Dynamics of the Consumption Share', 'FontSize', base_font_size, 'FontWeight', 'bold');             
        legend([ p1, p2, p3 ],  {'$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8'},  'Location', 'Northeast', 'Interpreter','LaTex', 'FontSize', 14) ; 
        %legend([ p1, p2, p3 ],  {'$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8'},  'Location', 'Northeast', 'Interpreter','LaTex', 'FontSize', 14) ; 
        ylim([-0.05; 0.045]);
        %axis( [ fmin fmax ymin  ymax ] );
        %set(gca,'XTick', [0; 0.1; 0.25; fCcovLE(2); 0.5; 0.65; 0.75; 0.9; 1]);            
        %set(gca,'XTickLabel',{'0', '0.1', '0.25', '0.35', '0.5', '0.65', '0.75', '0.9', '1'})        
        %set(gca,'YTick', [-0.007; fClamMinVal(3); -0.0025; fClamMinVal(2); 0; lamM(1,1); 0.0025; fClamMaxVal(2);  0.005;  0.0075; 0.01; 0.0125; fClamMaxVal(3);  0.015]);            
        %set(gca,'YTickLabel',{'-0.70', '-0.53', '-0.25', '-0.05', '0', '0.11', '0.25', '0.42', '0.50', '0.75', '1.00', '1.25', '1.35', '1.5'})        
        %set(gca,'YTickLabel',{'-70', '-53', '-25', '-5', '0', '11', '25', '42', '50', '75', '100', '125', '135', '150'})        
        grid on
        % ----------------------------------------------------------------------------------------------
end
%
    % -----------------------------------------------------------------------------------
if sl == 2          % Conditional Stock Market Volatility 
    % -----------------------------------------------------------------------------------
    nu = 0.02;
    kap = 0.01;
    lbar = 0; 
    sigl = 0.1;
    rhoA = 0;
    rhoB = 0.05;
    rho = mean([rhoA;rhoB]);
    drho = rhoB-rhoA;
    sigY = 0.033;
    muY = 0.02;
    %
    dim = 1000;
    fmin = 0;
    fmax = 1;
    ftVec = linspace(fmin, fmax,dim);
    DelVec = [0; 0.4; 0.8];
    dim2 = length(DelVec);
    %
    for i = 1:dim
        ft = ftVec(i);
        for j = 1:dim2
            Del = DelVec(j);
            [ ThetDem(i,j), sigMdem(i,j), lamM(i,j), sigM(i,j), PDt(i,j) ] = OnlineAppendixStockMarket( nu, rhoA, rhoB, ft, sigY, Del );
        end
    end
    %
    for j = 1:dim2
        Del = DelVec(j);
        [ ~, ~, ~, ~, ~, fCsig(j) ] = OnlineAppendixStockMarket( nu, rhoA, rhoB, 0.5, sigY, Del );
        [ ~, sigMdemC(j), ~, sigMC(j) ] = OnlineAppendixStockMarket( nu, rhoA, rhoB, fCsig(j), sigY, Del );
    end
    %
    base_font_size = 18;
    %
    
        %%
        figure;
        lines_str = {'-.r', '--b', 'k' };
        lines_str2 = {':r', ':b', ':k' };
        %
        ymin = 0; ymax = 0.25;
        %
        for j = 1:dim2
            plot(ftVec, sigM(:,j), lines_str{j}, 'LineWidth', 2);
            legendInfo{j} = ['$\Delta$ = ' num2str(DelVec(j)) ];
            hold on
        end
        %
        for j = 1:dim2
            plot([fCsig(j), fCsig(j)], [ymin, sigMC(j)], lines_str2{j}, [fmin, fCsig(j)], [sigMC(j), sigMC(j)], lines_str2{j}, 'LineWidth', 2);
            hold on
        end
        %
        legend( legendInfo, 'Location', 'Best', 'Interpreter','LaTex'); 
        %     
        xlabel('Consumption share, f_t', 'FontSize',base_font_size);
        ylabel('Stock Market Volatility', 'FontSize',base_font_size);
        %axis( [ sA sB ymin  ymax ] );
        %set(gca,'YTick', [ -0.25; 0; 0.25; 0.5; 0.75 ]);            
        %set(gca,'YTickLabel',{'-0.25', '0', '0.25', '0.5', '0.75'})               
        %set(gca,'XTick', [-3; -2.5; -2; -1.5; -1.25; -1;   -0.75; -0.5; 0]);            
        %set(gca,'XTickLabel',{'-3', '-2.5', '-2', '-1.5', '-1.25', '-1', '-0.75', '-0.5', '0'})        
        grid on
        %%
        figure;
        lines_str = {'-.r', '--b', 'k' };
        lines_str2 = {':r', ':b', ':k' };
        %
        ymin = 0; ymax = 0.26;
        %
        for j = 1:dim2
            plot(ftVec, sigMdem(:,j), lines_str{j}, 'LineWidth', 2);
            legendInfo{j} = ['$\Delta$ = ' num2str(DelVec(j)) ];
            hold on
        end
        %
        for j = 1:dim2
            plot([fCsig(j), fCsig(j)], [ymin, sigMdemC(j)], lines_str2{j}, [fmin, fCsig(j)], [sigMdemC(j), sigMdemC(j)], lines_str2{j}, 'LineWidth', 2);
            hold on
        end
        %
        legend( legendInfo, 'Location', 'Best', 'Interpreter','LaTex'); 
        %     
        xlabel('Consumption share, f_t', 'FontSize',base_font_size);
        ylabel('Stock Market Volatility', 'FontSize',base_font_size);
        %axis( [ sA sB ymin  ymax ] );
        %set(gca,'YTick', [ -0.25; 0; 0.25; 0.5; 0.75 ]);            
        %set(gca,'YTickLabel',{'-0.25', '0', '0.25', '0.5', '0.75'})               
        %set(gca,'XTick', [-3; -2.5; -2; -1.5; -1.25; -1;   -0.75; -0.5; 0]);            
        %set(gca,'XTickLabel',{'-3', '-2.5', '-2', '-1.5', '-1.25', '-1', '-0.75', '-0.5', '0'})        
        grid on
        %%
        ddim = 50;
        figure;
        p1 = plot(ftVec, sigM(:,1), 'k', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on;
        p2 = plot(ftVec, sigM(:,2), '--bo', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p3 = plot(ftVec, sigM(:,3), '-.rx', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p4 = plot([fCsig(3), fCsig(3)], [ymin, sigMC(3)], ':r', [fmin, fCsig(3)], [sigMC(3), sigMC(3)], ':r', 'LineWidth', 2);
        hold on
        p5 = plot([fCsig(2), fCsig(2)], [ymin, sigMC(2)], ':b', [fmin, fCsig(2)], [sigMC(2), sigMC(2)], ':b', 'LineWidth', 2);
        hold off
        %
        xlabel('Consumption Share $f_t$', 'Interpreter','LaTex', 'FontSize',base_font_size);
        %ylabel('Return Volatility $\sigma_{S,t}^{\alpha}$ in %', 'Interpreter','LaTex', 'FontSize', base_font_size );
        ylabel('Return Volatility ($\%$)', 'Interpreter','LaTex', 'FontSize', base_font_size );
        title('Stock Market', 'FontSize', base_font_size, 'FontWeight', 'bold');             
        legend([ p1, p2, p3 ],  {'$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8'},  'Location', 'Northeast', 'Interpreter','LaTex', 'FontSize', 14) ; 
        %legend([ p1, p2, p3 ],  {'$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8'},  'Location', 'Northeast', 'Interpreter','LaTex', 'FontSize', 14) ; 
        axis( [ fmin fmax ymin  ymax ] );
        set(gca,'XTick', [0; 0.1; 0.2; 0.3; fCsig(2); 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1]);            
        set(gca,'XTickLabel',{'0', '0.1', '0.2', '0.3', '0.35', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1'})        
        set(gca,'YTick', [0; sigMC(1); 0.05; 0.1; sigMC(2); 0.15; 0.2; sigMC(3); 0.26]);            
        set(gca,'YTickLabel',{'0', '3.3', '5.0', '10.0', '12.2', '15.0', '20.0', '23.6', '26.0'})        
        grid on
        %%
        %
end    
        
    % -----------------------------------------------------------------------------------
if sl == 3          % Conditional Stock Market Risk Premium
    % -----------------------------------------------------------------------------------
    nu = 0.02;
    kap = 0.01;
    lbar = 0; 
    sigl = 0.1;
    rhoA = 0.001;
    rhoB = 0.05;
    rho = mean([rhoA;rhoB]);
    drho = rhoB-rhoA;
    sigY = 0.033;
    muY = 0.02;
    %
    dim = 1000;
    fmin = 0;
    fmax = 1;
    ftVec = linspace(fmin, fmax,dim);
    DelVec = [0; 0.4; 0.8];
    dim2 = length(DelVec);
    %
    for i = 1:dim
        ft = ftVec(i);
        for j = 1:dim2
            Del = DelVec(j);
            [ ThetDem(i,j), ~, lamM(i,j), ~, PDt(i,j) ] = OnlineAppendixStockMarket( nu, rhoA, rhoB, ft, sigY, Del );
        end
    end
    %
    for j = 1:dim2
        Del = DelVec(j);
        [ fClamMax(j), fClamMaxVal(j), fClamMin(j), fClamMinVal(j) ] = OnlineAppendixStockMarketCriticalValues( nu, rhoA, rhoB, sigY, Del );
    end
    %
    base_font_size = 18;
    %
    
        %%
        figure;
        lines_str = {'-.r', '--b', 'k' };
        lines_str2 = {':r', ':b', ':k' };
        %
        ymin = 0; ymax = 0.25;
        %
        for j = 1:dim2
            plot(ftVec, lamM(:,j), lines_str{j}, 'LineWidth', 2);
            legendInfo{j} = ['$\Delta$ = ' num2str(DelVec(j)) ];
            hold on
        end
        %
        for j = 1:dim2
            plot([fClamMax(j), fClamMax(j)], [ymin, fClamMaxVal(j)], lines_str2{j}, [fmin, fClamMax(j)], [fClamMaxVal(j), fClamMaxVal(j)], lines_str2{j}, 'LineWidth', 2);
            hold on
        end
        %
        legend( legendInfo, 'Location', 'Best', 'Interpreter','LaTex'); 
        %     
        xlabel('Consumption share, f_t', 'FontSize',base_font_size);
        ylabel('Stock Market Risk Premium', 'FontSize',base_font_size);
        %axis( [ sA sB ymin  ymax ] );
        %set(gca,'YTick', [ -0.25; 0; 0.25; 0.5; 0.75 ]);            
        %set(gca,'YTickLabel',{'-0.25', '0', '0.25', '0.5', '0.75'})               
        %set(gca,'XTick', [-3; -2.5; -2; -1.5; -1.25; -1;   -0.75; -0.5; 0]);            
        %set(gca,'XTickLabel',{'-3', '-2.5', '-2', '-1.5', '-1.25', '-1', '-0.75', '-0.5', '0'})        
        grid on
        %%
        figure;
        lines_str = {'-.r', '--b', 'k' };
        lines_str2 = {':r', ':b', ':k' };
        %
        ymin = -0.03; ymax = 0.06;
        %
        for j = 1:dim2
            plot(ftVec, ThetDem(:,j), lines_str{j}, 'LineWidth', 2);
            legendInfo{j} = ['$\Delta$ = ' num2str(DelVec(j)) ];
            hold on
        end
        %
        %
        legend( legendInfo, 'Location', 'Best', 'Interpreter','LaTex'); 
        %     
        xlabel('Consumption Share $f_t$', 'Interpreter','LaTex', 'FontSize',base_font_size);
        ylabel('Market Price of Demand Risk', 'FontSize',base_font_size);
        %axis( [ sA sB ymin  ymax ] );
        %set(gca,'YTick', [ -0.25; 0; 0.25; 0.5; 0.75 ]);            
        %set(gca,'YTickLabel',{'-0.25', '0', '0.25', '0.5', '0.75'})               
        %set(gca,'XTick', [-3; -2.5; -2; -1.5; -1.25; -1;   -0.75; -0.5; 0]);            
        %set(gca,'XTickLabel',{'-3', '-2.5', '-2', '-1.5', '-1.25', '-1', '-0.75', '-0.5', '0'})        
        grid on
        %%
        ddim = 50;
        figure;
        p1 = plot(ftVec, lamM(:,1), 'k', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on;
        p2 = plot(ftVec, lamM(:,2), '--bo', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p3 = plot(ftVec, lamM(:,3), '-.rx', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p4 = plot([fClamMax(2), fClamMax(2)], [sigY^2, fClamMaxVal(2)], ':b', [fmin, fClamMax(2)], [fClamMaxVal(2), fClamMaxVal(2)], ':b', 'LineWidth', 2);
        hold on
        p5 = plot([fClamMax(3), fClamMax(3)], [fClamMaxVal(2), fClamMaxVal(3)], ':r', [fmin, fClamMax(3)], [fClamMaxVal(3), fClamMaxVal(3)], ':r', 'LineWidth', 2);
        hold on
        p6 = plot([fClamMin(2), fClamMin(2)], [fClamMinVal(2), sigY^2], ':b', [fmin, fClamMin(2)], [fClamMinVal(2), fClamMinVal(2)], ':b', 'LineWidth', 2);
        hold on
        p7 = plot([fClamMin(3), fClamMin(3)], [fClamMinVal(3), fClamMinVal(2)], ':r', [fmin, fClamMin(3) ], [fClamMinVal(3), fClamMinVal(3)], ':r', 'LineWidth', 2);
        hold off
        %
        xlabel('Consumption Share $f_t$', 'Interpreter','LaTex', 'FontSize',base_font_size);
        %ylabel('Return Volatility $\sigma_{S,t}^{\alpha}$ in %', 'Interpreter','LaTex', 'FontSize', base_font_size );
        %ylabel('Risk Premium (%)', 'Interpreter','LaTex', 'FontSize', base_font_size );
        ylabel('Risk Premium ($\%$)', 'Interpreter','LaTex', 'FontSize', base_font_size );
        title('Stock Market', 'FontSize', base_font_size, 'FontWeight', 'bold');             
        legend([ p1, p2, p3 ],  {'$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8'},  'Location', 'Northeast', 'Interpreter','LaTex', 'FontSize', 14) ; 
        %legend([ p1, p2, p3 ],  {'$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8'},  'Location', 'Northeast', 'Interpreter','LaTex', 'FontSize', 14) ; 
        axis( [ fmin fmax ymin  ymax ] );
        set(gca,'XTick', [0; 0.1; fClamMax(2); 0.25; 0.3; 0.4; 0.5; 0.6; 0.7; fClamMin(2); 0.85; 0.9; 1]);            
        set(gca,'XTickLabel',{'0', '0.1', '0.17', '0.25', '0.3', '0.4', '0.5', '0.6', '0.7', '0.76', '0.85', '0.9', '1'})        
        set(gca,'YTick', [fClamMinVal(3); -0.015; fClamMinVal(2); lamM(1,1); fClamMaxVal(2);  0.025;  0.04; fClamMaxVal(3)]);            
        %set(gca,'YTickLabel',{'-0.70', '-0.53', '-0.25', '-0.05', '0', '0.11', '0.25', '0.42', '0.50', '0.75', '1.00', '1.25', '1.35', '1.5'})        
        set(gca,'YTickLabel',{'-2.45', '-1.50', '-0.53', '0.11', '1.35', '2.50', '4.00', '5.09'})        
        grid on
        %%
        %
        %
end

  
%
    % -----------------------------------------------------------------------------------
if sl == 4          % Trading volume as a function of the consumption share
    % -----------------------------------------------------------------------------------
    nu = 0.02;
    kap = 0.01;
    lbar = 0; 
    sigl = 0.1;
    rhoA = 0.001;
    rhoB = 0.05;
    rho = mean([rhoA;rhoB]);
    drho = rhoB-rhoA;
    sigY = 0.033;
    muY = 0.02;
    %
    dim = 1000;
    fmin = 0;
    fmax = 1;
    ftVec = linspace(fmin, fmax,dim);
    DelVec = [0; 0.4; 0.8];
    dim2 = length(DelVec);
    %
    for i = 1:dim
        ft = ftVec(i);
        for j = 1:dim2
            Del = DelVec(j);
            [ ~, ~, ~, ~, ~, ~, ~, ~, XEalfa(i,j), TVol(i,j) ] = OnlineAppendixStockMarket( nu, rhoA, rhoB, ft, sigY, Del );
        end
    end
    %
    for j = 1:dim2
        Del = DelVec(j);
        [ ~, ~, ~, ~, ~, fCsig(j) ] = OnlineAppendixStockMarket( nu, rhoA, rhoB, 0.5, sigY, Del );
        [ ~, ~, ~, ~, ~, ~, ~, ~, XEalfaC(j),  TVolC(j)] = OnlineAppendixStockMarket( nu, rhoA, rhoB, fCsig(j), sigY, Del );
    end
    %
    %
    base_font_size = 18;
    %
    
        %%
        figure;
        lines_str = {'-.r', '--b', 'k' };
        lines_str2 = {':r', ':b', ':k' };
        %
        ymin = 0; ymax = 0.25;
        %
        for j = 1:dim2
            plot(ftVec, XEalfa(:,j), lines_str{j}, 'LineWidth', 2);
            legendInfo{j} = ['$\Delta$ = ' num2str(DelVec(j)) ];
            hold on
        end
        %
        %
        for j = 1:dim2
            plot([fCsig(j), fCsig(j)], [ymin, XEalfaC(j)], lines_str2{j}, [fmin, fCsig(j)], [XEalfaC(j), XEalfaC(j)], lines_str2{j}, 'LineWidth', 2);
            hold on
        end
        %
        legend( legendInfo, 'Location', 'Best', 'Interpreter','LaTex'); 
        %     
        xlabel('Consumption share, f_t', 'FontSize',base_font_size);
        ylabel('Excess Exposure to Demand Shocks', 'FontSize',base_font_size);
        %axis( [ sA sB ymin  ymax ] );
        %set(gca,'YTick', [ -0.25; 0; 0.25; 0.5; 0.75 ]);            
        %set(gca,'YTickLabel',{'-0.25', '0', '0.25', '0.5', '0.75'})               
        %set(gca,'XTick', [-3; -2.5; -2; -1.5; -1.25; -1;   -0.75; -0.5; 0]);            
        %set(gca,'XTickLabel',{'-3', '-2.5', '-2', '-1.5', '-1.25', '-1', '-0.75', '-0.5', '0'})        
        grid on
        %%
        %
        %
        %
        %%
        figure;
        lines_str = {'-.r', '--b', 'k' };
        lines_str2 = {':r', ':b', ':k' };
        %
        ymin = 0; ymax = 0.25;
        %
        for j = 1:dim2
            plot(ftVec, TVol(:,j), lines_str{j}, 'LineWidth', 2);
            legendInfo{j} = ['$\Delta$ = ' num2str(DelVec(j)) ];
            hold on
        end
        %
        %
        for j = 1:dim2
            plot([fCsig(j), fCsig(j)], [ymin, TVolC(j)], lines_str2{j}, [fmin, fCsig(j)], [TVolC(j), TVolC(j)], lines_str2{j}, 'LineWidth', 2);
            hold on
        end
        %
        legend( legendInfo, 'Location', 'Best', 'Interpreter','LaTex'); 
        %     
        xlabel('Consumption share, f_t', 'FontSize',base_font_size);
        ylabel('Excess Exposure to Demand Shocks', 'FontSize',base_font_size);
        %axis( [ sA sB ymin  ymax ] );
        %set(gca,'YTick', [ -0.25; 0; 0.25; 0.5; 0.75 ]);            
        %set(gca,'YTickLabel',{'-0.25', '0', '0.25', '0.5', '0.75'})               
        %set(gca,'XTick', [-3; -2.5; -2; -1.5; -1.25; -1;   -0.75; -0.5; 0]);            
        %set(gca,'XTickLabel',{'-3', '-2.5', '-2', '-1.5', '-1.25', '-1', '-0.75', '-0.5', '0'})        
        grid on
        %%
        %
        %
        %
        %%
        ymin = -0.01; ymax = 0.075;
        ddim = 50;
        figure;
        p1 = plot(ftVec, TVol(:,1), '-.rx', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on;
        p2 = plot(ftVec, TVol(:,2), '--bo', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p3 = plot(ftVec, TVol(:,3), 'k', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p4 = plot([fCsig(2), fCsig(2)], [ymin, TVolC(2)], ':k', [fmin, fCsig(2)], [TVolC(2), TVolC(2)], ':k', 'LineWidth', 2);
        hold on
        p5 = plot([fCsig(3), fCsig(3)], [ymin, TVolC(3)], ':k', [fmin, fCsig(3)], [TVolC(3), TVolC(3)], ':k', 'LineWidth', 2);
        hold off
        %
       xlabel('Consumption Share $f_t$', 'Interpreter','LaTex', 'FontSize',base_font_size);
        %ylabel('Return Volatility $\sigma_{S,t}^{\alpha}$ in %', 'Interpreter','LaTex', 'FontSize', base_font_size );
        ylabel('Trading Volume', 'Interpreter','LaTex', 'FontSize', base_font_size );
        title('Stock Market', 'FontSize', base_font_size, 'FontWeight', 'bold');             
        legend([ p1, p2, p3 ],  {'$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8'},  'Location', 'Northeast', 'Interpreter','LaTex', 'FontSize', 14) ; 
        %legend([ p1, p2, p3 ],  {'$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8'},  'Location', 'Northeast', 'Interpreter','LaTex', 'FontSize', 14) ; 
        axis( [ fmin fmax ymin  ymax ] );
        set(gca,'XTick', [0; 0.1; 0.2; 0.3; 0.3539; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1]);            
        set(gca,'XTickLabel',{'0', '0.1', '0.2', '0.3', '0.35', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1'})        
        %set(gca,'YTick', [0; sigMC(1); 0.05; 0.1; sigMC(2); 0.15; 0.2; sigMC(3); 0.25]);            
        %set(gca,'YTickLabel',{'0', '3.3', '5.0', '10.0', '12.2', '15.0', '20.0', '23.6', '25.0'})        
        grid on
        %%
        %
        %
end    

    % -----------------------------------------------------------------------------------
if sl == 5         % Stock Market Return Predictability  as a function of the consumption share ft 
    % -----------------------------------------------------------------------------------
    nu = 0.02;
    kap = 0.01;
    lbar = 0; 
    sigl = 0.1;
    rhoA = 0.001;
    rhoB = 0.05;
    rho = mean([rhoA;rhoB]);
    drho = rhoB-rhoA;
    sigY = 0.033;
    muY = 0.02;
    %
    dim = 1000;
    fmin = 0;
    fmax = 1;
    ftVec = linspace(fmin, fmax,dim);
    DelVec = [0; 0.4; 0.8];
    dim2 = length(DelVec);
    %
    for i = 1:dim
        ft = ftVec(i);
        for j = 1:dim2
            Del = DelVec(j);
            [ ThetDem(i,j), ~, lamM(i,j), ~, PDt(i,j) ] = OnlineAppendixStockMarket( nu, rhoA, rhoB, ft, sigY, Del );
        end
    end
    %
    for j = 1:dim2
        Del = DelVec(j);
        [ fClamMax(j), fClamMaxVal(j), fClamMin(j), fClamMinVal(j) ] = OnlineAppendixStockMarketCriticalValues( nu, rhoA, rhoB, sigY, Del );
    end
    %
    base_font_size = 18;
    %
    
        %%
        figure;
        lines_str = {'-.r', '--b', 'k' };
        lines_str2 = {':r', ':b', ':k' };
        %
        ymin = 0; ymax = 0.25;
        %
        for j = 1:dim2
            plot(ftVec, lamM(:,j), lines_str{j}, 'LineWidth', 2);
            legendInfo{j} = ['$\Delta$ = ' num2str(DelVec(j)) ];
            hold on
        end
        %
        for j = 1:dim2
            plot([fClamMax(j), fClamMax(j)], [ymin, fClamMaxVal(j)], lines_str2{j}, [fmin, fClamMax(j)], [fClamMaxVal(j), fClamMaxVal(j)], lines_str2{j}, 'LineWidth', 2);
            hold on
        end
        %
        legend( legendInfo, 'Location', 'Best', 'Interpreter','LaTex'); 
        %     
        xlabel('Consumption share, f_t', 'FontSize',base_font_size);
        ylabel('Stock Market Risk Premium', 'FontSize',base_font_size);
        %axis( [ sA sB ymin  ymax ] );
        %set(gca,'YTick', [ -0.25; 0; 0.25; 0.5; 0.75 ]);            
        %set(gca,'YTickLabel',{'-0.25', '0', '0.25', '0.5', '0.75'})               
        %set(gca,'XTick', [-3; -2.5; -2; -1.5; -1.25; -1;   -0.75; -0.5; 0]);            
        %set(gca,'XTickLabel',{'-3', '-2.5', '-2', '-1.5', '-1.25', '-1', '-0.75', '-0.5', '0'})        
        grid on
        %%
        figure;
        lines_str = {'-.r', '--b', 'k' };
        lines_str2 = {':r', ':b', ':k' };
        %
        ymin = -0.03; ymax = 0.06;
        %
        for j = 1:dim2
            plot(ftVec, ThetDem(:,j), lines_str{j}, 'LineWidth', 2);
            legendInfo{j} = ['$\Delta$ = ' num2str(DelVec(j)) ];
            hold on
        end
        %
        %
        legend( legendInfo, 'Location', 'Best', 'Interpreter','LaTex'); 
        %     
        xlabel('Consumption share, f_t', 'FontSize',base_font_size);
        ylabel('Market Price of Demand Risk', 'FontSize',base_font_size);
        %axis( [ sA sB ymin  ymax ] );
        %set(gca,'YTick', [ -0.25; 0; 0.25; 0.5; 0.75 ]);            
        %set(gca,'YTickLabel',{'-0.25', '0', '0.25', '0.5', '0.75'})               
        %set(gca,'XTick', [-3; -2.5; -2; -1.5; -1.25; -1;   -0.75; -0.5; 0]);            
        %set(gca,'XTickLabel',{'-3', '-2.5', '-2', '-1.5', '-1.25', '-1', '-0.75', '-0.5', '0'})        
        grid on
        %%
        ddim = 50;
        figure;
        p1 = plot(ftVec, lamM(:,1), '-.rx', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on;
        p2 = plot(ftVec, lamM(:,2), '--bo', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p3 = plot(ftVec, lamM(:,3), 'k', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p4 = plot([fClamMax(2), fClamMax(2)], [ymin, fClamMaxVal(2)], ':k', [fmin, fClamMax(2)], [fClamMaxVal(2), fClamMaxVal(2)], ':k', 'LineWidth', 2);
        hold on
        p5 = plot([fClamMax(3), fClamMax(3)], [ymin, fClamMaxVal(3)], ':k', [fmin, fClamMax(3)], [fClamMaxVal(3), fClamMaxVal(3)], ':k', 'LineWidth', 2);
        hold on
        p6 = plot([fClamMin(2), fClamMin(2)], [ymin, fClamMinVal(2)], ':k', [fmin, fClamMin(2)], [fClamMinVal(2), fClamMinVal(2)], ':k', 'LineWidth', 2);
        hold on
        p7 = plot([fClamMin(3), fClamMin(3)], [ymin, fClamMinVal(3)], ':k', [fmin, fClamMin(3)], [fClamMinVal(3), fClamMinVal(3)], ':k', 'LineWidth', 2);
        hold off
        %
        xlabel('Consumption Share f_t', 'FontSize',base_font_size);
        %ylabel('Return Volatility $\sigma_{S,t}^{\alpha}$ in %', 'Interpreter','LaTex', 'FontSize', base_font_size );
        %ylabel('Risk Premium (%)', 'Interpreter','LaTex', 'FontSize', base_font_size );
        ylabel('Risk Premium (%)', 'Interpreter','LaTex', 'FontSize', base_font_size );
        title('Stock Market', 'FontSize', base_font_size, 'FontWeight', 'bold');             
        legend([ p1, p2, p3 ],  {'$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8'},  'Location', 'Northeast', 'Interpreter','LaTex', 'FontSize', 14) ; 
        %legend([ p1, p2, p3 ],  {'$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8'},  'Location', 'Northeast', 'Interpreter','LaTex', 'FontSize', 14) ; 
        axis( [ fmin fmax ymin  ymax ] );
        set(gca,'XTick', [0; 0.1; fClamMax(2); 0.25; 0.3; 0.4; 0.5; 0.6; 0.7; fClamMin(2); 0.85; 0.9; 1]);            
        set(gca,'XTickLabel',{'0', '0.1', '0.17', '0.25', '0.3', '0.4', '0.5', '0.6', '0.7', '0.76', '0.85', '0.9', '1'})        
        set(gca,'YTick', [fClamMinVal(3); -0.015; fClamMinVal(2); lamM(1,1); fClamMaxVal(2);  0.025;  0.04; fClamMaxVal(3)]);            
        %set(gca,'YTickLabel',{'-0.70', '-0.53', '-0.25', '-0.05', '0', '0.11', '0.25', '0.42', '0.50', '0.75', '1.00', '1.25', '1.35', '1.5'})        
        set(gca,'YTickLabel',{'-2.45', '-1.50', '-0.53', '0.11', '1.35', '2.50', '4.00', '5.09'})        
        grid on
        %%
        %
        %
        %
        %
        base_font_size = 18;
        ymin = -1.1; ymax = 1.1;
        %
        fClamMax = fClamMax(2);
        fClamMin = fClamMin(2);
        %
        X1 = linspace(0,fClamMax,dim);          
        X2 = linspace(fClamMax,fClamMin,dim);      
        X3 = linspace(fClamMin,1,dim);
        %
        ddim = 100;
        figure;
        p1 = plot(ftVec, zeros(1,dim), '-.rx', 'MarkerIndices', 1:ddim/2:dim, 'LineWidth', 2);
        hold on;
        p2 = plot(X1, ones(1,dim), '--bo', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p3 = plot(X2, -ones(1,dim), '--bo', 'MarkerIndices', 1:ddim/2:dim, 'LineWidth', 2);
        hold on
        p4 = plot(X3, ones(1,dim), '--bo', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p5 = plot(X1, ones(1,dim), 'k', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p6 = plot(X2, -ones(1,dim), 'k', 'MarkerIndices', 1:ddim/2:dim, 'LineWidth', 2);
        hold on
        p7 = plot(X3, ones(1,dim), 'k', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p8 = plot([fClamMax, fClamMax], [-1, 1], ':k', [fClamMin, fClamMin], [-1, 1], ':k', 'LineWidth', 2);
        hold on
        plot(fClamMax,0,'bo', fClamMin,0,'bo', fClamMax,1,'bo', fClamMin,-1,'bo')
        hold off
        %
        xlabel('Consumption Share f_t', 'FontSize',base_font_size);
        ylabel('Local Correlation $\rho_{PD}$', 'Interpreter','LaTex', 'FontSize', base_font_size );
        title('Stock Market Predictability', 'FontSize', base_font_size, 'FontWeight', 'bold');             
        legend([ p1, p2, p5 ],  {'$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8'},  'Location', 'Northeast', 'Interpreter','LaTex', 'FontSize', 14) ; 
        axis( [ fmin fmax ymin  ymax ] );
        set(gca,'XTick', [0; 0.1; fClamMax; 0.25; 0.3; 0.4; 0.5; 0.6; 0.7; fClamMin; 0.85; 0.9; 1]);            
        set(gca,'XTickLabel',{'0', '0.1', '0.17', '0.25', '0.3', '0.4', '0.5', '0.6', '0.7', '0.76', '0.85', '0.9', '1'})        
        %set(gca,'YTick', [fClamMinVal(3); -0.015; fClamMinVal(2); lamM(1,1); fClamMaxVal(2);  0.025;  0.04; fClamMaxVal(3)]);            
        %set(gca,'YTickLabel',{'-0.70', '-0.53', '-0.25', '-0.05', '0', '0.11', '0.25', '0.42', '0.50', '0.75', '1.00', '1.25', '1.35', '1.5'})        
        %set(gca,'YTickLabel',{'-2.45', '-1.50', '-0.53', '0.11', '1.35', '2.50', '4.00', '5.09'})        
        grid on
        
        
        
end

    % -------------------------------------------------------------------------------------------------------------------
if sl == 6          % Leverage Effect for Stock Market Returns as a function of the consumption share ft  
    % -------------------------------------------------------------------------------------------------------------------
    nu = 0.02;
    kap = 0.01;
    lbar = 0; 
    sigl = 0.1;
    rhoA = 0.001;
    rhoB = 0.05;
    rho = mean([rhoA;rhoB]);
    drho = rhoB-rhoA;
    sigY = 0.033;
    muY = 0.02;
    %
    dim = 1000;
    fmin = 0;
    fmax = 1;
    ftVec = linspace(fmin, fmax,dim);
    DelVec = [0; 0.4; 0.8];
    dim2 = length(DelVec);
    %
    [ ~, ~, ~, ~, ~, ~, ~, fCrhoLE ] = OnlineAppendixStockMarket( nu, rhoA, rhoB, 0.5, sigY, DelVec(2) );
    %
    for i = 1:dim
        ft = ftVec(i);
        for j = 1:dim2
            Del = DelVec(j);
            [ ~, ~, ~, ~, ~, ~, valF ] = OnlineAppendixStockMarket( nu, rhoA, rhoB, ft, sigY, Del );
            if ft < fCrhoLE
                rhoLEpos(i,j) = valF;
                rhoLEneg(i,j) = NaN;
            elseif ft > fCrhoLE
                rhoLEpos(i,j) = NaN;
                rhoLEneg(i,j) = valF;
            else
                rhoLEpos(i,j) = 0;
                rhoLEneg(i,j) = 0;
            end
        end
    end
    %
    base_font_size = 18;
    %
    
        %%
        figure;
        lines_str = {'-.r', '--b', 'k' };
        %
        ymin = -1; ymax = 1;
        %
        for j = 1:dim2
            plot(ftVec, rhoLEpos(:,j), lines_str{j}, 'LineWidth', 2);
            hold on 
            plot(ftVec, rhoLEneg(:,j), lines_str{j}, 'LineWidth', 2);
            hold on
            legendInfo{j} = ['$\Delta$ = ' num2str(DelVec(j)) ];
        end
        %
        %for j = 1:dim2
        %    plot([fCsig(j), fCsig(j)], [ymin, sigMC(j)], lines_str2{j}, [fmin, fCsig(j)], [sigMC(j), sigMC(j)], lines_str2{j}, 'LineWidth', 2);
        %    hold on
        %end
        %
        legend( legendInfo, 'Location', 'Best', 'Interpreter','LaTex'); 
        %     
        xlabel('Consumption share, f_t', 'FontSize',base_font_size);
        ylabel('Leverage Effect', 'FontSize',base_font_size);
        %axis( [ sA sB ymin  ymax ] );
        %set(gca,'YTick', [ -0.25; 0; 0.25; 0.5; 0.75 ]);            
        %set(gca,'YTickLabel',{'-0.25', '0', '0.25', '0.5', '0.75'})               
        %set(gca,'XTick', [-3; -2.5; -2; -1.5; -1.25; -1;   -0.75; -0.5; 0]);            
        %set(gca,'XTickLabel',{'-3', '-2.5', '-2', '-1.5', '-1.25', '-1', '-0.75', '-0.5', '0'})        
        grid on
        %%
        ddim = 50;
        figure;
        p1 = plot(ftVec, rhoLEpos(:,1), 'k', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on;
        p2 = plot(ftVec, rhoLEneg(:,1), 'k', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on;
        p3 = plot(ftVec, rhoLEpos(:,2), '--bo', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on;
        p4 = plot(ftVec, rhoLEneg(:,2), '--bo', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on;
        p5 = plot(ftVec, rhoLEpos(:,3), '-.rx', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on;
        p6 = plot(ftVec, rhoLEneg(:,3), '-.rx', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on;
        p7 = plot([fCrhoLE, fCrhoLE], [ymin, ymax], ':k', 'LineWidth', 2);
        hold on
        plot(fCrhoLE,0,'bo');
        hold off
        %
        xlabel('Consumption Share $f_t$', 'Interpreter','LaTex', 'FontSize',base_font_size);
        ylabel('Local Correlation $\rho_{LE}$', 'Interpreter','LaTex', 'FontSize', base_font_size );
        title('Black''s "Leverage" Effect', 'FontSize', base_font_size, 'FontWeight', 'bold');             
        legend([ p1, p3, p5 ],  {'$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8'},  'Location', 'Northeast', 'Interpreter','LaTex', 'FontSize', 14) ; 
        %legend([ p1, p2, p3 ],  {'$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8'},  'Location', 'Northeast', 'Interpreter','LaTex', 'FontSize', 14) ; 
        axis( [ fmin fmax -1.1  1.1 ] );
        set(gca,'XTick', [0; 0.1; 0.2; 0.3; fCrhoLE; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1]);            
        set(gca,'XTickLabel',{'0', '0.1', '0.2', '0.3', '0.35', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1'})        
        %set(gca,'YTick', [0; sigMC(1); 0.05; 0.1; sigMC(2); 0.15; 0.2; sigMC(3); 0.25]);            
        %set(gca,'YTickLabel',{'0', '3.3', '5.0', '10.0', '12.2', '15.0', '20.0', '23.6', '25.0'})        
        grid on
        %%
        %
        %
end    
 
if sl == 7
   %
   %load('JFround1MatlabUnconditionalStatsBetterFINAL.mat');
   load('OnlineAppendixMatlabUnconditionalStatsBetter.mat');
   %load('Data/Model Disagreement/MatFiles100Kyears/ConsumptionShareBaseModel.mat', 'DelVec', 'fmean', 'fvol', 'fskew', 'fkurt');
   plotf = figure;
   hold on
   [ax2, hl2, hr2] = plotyy(DelVec, fmean, DelVec, fvol);
   set(hl2, 'Color', 'b', 'LineWidth', 2, 'LineStyle', '--');
   set(hr2, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '-.');
   xlabel('Disagreement ($\Delta$)', 'FontSize', base_font_size, 'Interpreter','LaTex'); 
   set(get(ax2(1),'Ylabel'),'String','Unconditional Mean', 'FontSize', base_font_size);
   set(get(ax2(2),'Ylabel'),'String','Unconditional Volatility', 'FontSize', base_font_size);
   title('Consumption Share', 'FontSize', base_font_size, 'Interpreter','LaTex', 'FontWeight', 'bold');
   ylim(ax2(1),[0.4,0.7]);
   ylim(ax2(2),[0,0.4]);
   %set(get(ax2(1),'XTick'), {'0','0.1','0.2','0.3','0.4', '0.5', '0.6','0.7','0.8','0.9','1'});            
   %set(gca,'XTickLabel',{'0', '0.1', '0.2', '0.3', '0.35', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1'})        
   %set(gca,'YTick', [0; sigMC(1); 0.05; 0.1; sigMC(2); 0.15; 0.2; sigMC(3); 0.25]);            
   %set(gca,'YTickLabel',{'0', '3.3', '5.0', '10.0', '12.2', '15.0', '20.0', '23.6', '25.0'})        
        
   grid on;
   hold off
   %
   %load('Data/Model Disagreement/MatFiles100Kyears/PriceDividendRatioBaseModel.mat', 'DelVec', 'PDmean', 'PDvol', 'PDskew', 'PDkurt');
   plotPD = figure;
   hold on
   [ax2, hl2, hr2] = plotyy(DelVec, PDmean, DelVec, PDvol);
   set(hl2, 'Color', 'b', 'LineWidth', 2, 'LineStyle', '--');
   set(hr2, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '-.');
   xlabel('Disagreement ($\Delta$)', 'FontSize', base_font_size, 'Interpreter','LaTex'); 
   set(get(ax2(1),'Ylabel'),'String','Unconditional Mean', 'FontSize', base_font_size);
   set(get(ax2(2),'Ylabel'),'String','Unconditional Volatility', 'FontSize', base_font_size);
   title('Wealth-Consumption Ratio', 'FontSize', base_font_size, 'Interpreter','LaTex', 'FontWeight', 'bold');
   grid on;
   hold off
   %load('Data/Model Disagreement/MatFiles100Kyears/RiskFreeRateBaseModel.mat', 'DelVec', 'rfmean', 'rfvol', 'rfskew', 'rfkurt');
   plotrf = figure;
   hold on
   [ax2, hl2, hr2] = plotyy(DelVec, rfmean, DelVec, rfvol);
   set(hl2, 'Color', 'b', 'LineWidth', 2, 'LineStyle', '--');
   set(hr2, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '-.');
   xlabel('Disagreement ($\Delta$)', 'FontSize', base_font_size, 'Interpreter','LaTex'); 
   set(get(ax2(1),'Ylabel'),'String','Unconditional Mean', 'FontSize', base_font_size);
   set(get(ax2(2),'Ylabel'),'String','Unconditional Volatility', 'FontSize', base_font_size);
   title('Risk-Free Interest Rate', 'FontSize', base_font_size, 'Interpreter','LaTex', 'FontWeight', 'bold');
   grid on;
   hold off
   %load('Data/Model Disagreement/MatFiles100Kyears/MarketPriceofDemandRiskBaseModel.mat', 'DelVec', 'ThetDemmean', 'ThetDemvol', 'ThetDemskew', 'ThetDemkurt');
   plotThetDem = figure;
   hold on
   [ax2, hl2, hr2] = plotyy(DelVec, ThetDemmean, DelVec, ThetDemvol);
   set(hl2, 'Color', 'b', 'LineWidth', 2, 'LineStyle', '--');
   set(hr2, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '-.');
   xlabel('Disagreement ($\Delta$)', 'FontSize', base_font_size, 'Interpreter','LaTex'); 
   set(get(ax2(1),'Ylabel'),'String','Unconditional Mean', 'FontSize', base_font_size);
   set(get(ax2(2),'Ylabel'),'String','Unconditional Volatility', 'FontSize', base_font_size);
   title('Market Price of Demand Shock Risk', 'FontSize', base_font_size, 'Interpreter','LaTex', 'FontWeight', 'bold');
   grid on;
   hold off
   %load('Data/Model Disagreement/MatFiles100Kyears/StockMarketExposure2DemandRiskBaseModel.mat', 'DelVec', 'sigSdemmean', 'sigSdemvol', 'sigSdemskew', 'sigSdemkurt');
   plotSigSdem = figure;
   hold on
   [ax2, hl2, hr2] = plotyy(DelVec, sigSdemmean, DelVec, sigSdemvol);
   set(hl2, 'Color', 'b', 'LineWidth', 2, 'LineStyle', '--');
   set(hr2, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '-.');
   xlabel('Disagreement ($\Delta$)', 'FontSize', base_font_size, 'Interpreter','LaTex'); 
   set(get(ax2(1),'Ylabel'),'String','Unconditional Mean', 'FontSize', base_font_size);
   set(get(ax2(2),'Ylabel'),'String','Unconditional Volatility', 'FontSize', base_font_size);
   title('Demand Shock Risk Volatility', 'FontSize', base_font_size, 'Interpreter','LaTex', 'FontWeight', 'bold');
   grid on;
   hold off
   %load('Data/Model Disagreement/MatFiles100Kyears/StockMarketVolatilityBaseModel.mat', 'DelVec', 'sigSmean', 'sigSvol', 'sigSskew', 'sigSkurt');
   plotSigS = figure;
   hold on
   [ax2, hl2, hr2] = plotyy(DelVec, sigSmean, DelVec, sigSvol);
   set(hl2, 'Color', 'b', 'LineWidth', 2, 'LineStyle', '--');
   set(hr2, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '-.');
   xlabel('Disagreement ($\Delta$)', 'FontSize', base_font_size, 'Interpreter','LaTex'); 
   set(get(ax2(1),'Ylabel'),'String','Unconditional Mean', 'FontSize', base_font_size);
   set(get(ax2(2),'Ylabel'),'String','Unconditional Volatility', 'FontSize', base_font_size);
   title('Stock Market Volatility', 'FontSize', base_font_size, 'Interpreter','LaTex', 'FontWeight', 'bold');
   grid on;
   hold off
   %load('Data/Model Disagreement/MatFiles100Kyears/StockMarketRiskPremiumBaseModel.mat', 'DelVec', 'lamSmean', 'lamSvol', 'lamSskew', 'lamSkurt');
   plotLamS = figure;
   hold on
   [ax2, hl2, hr2] = plotyy(DelVec, lamSmean, DelVec, lamSvol);
   set(hl2, 'Color', 'b', 'LineWidth', 2, 'LineStyle', '--');
   set(hr2, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '-.');
   xlabel('Disagreement ($\Delta$)', 'FontSize', base_font_size, 'Interpreter','LaTex'); 
   set(get(ax2(1),'Ylabel'),'String','Unconditional Mean', 'FontSize', base_font_size);
   set(get(ax2(2),'Ylabel'),'String','Unconditional Volatility', 'FontSize', base_font_size);
   title('Stock Market Risk Premium', 'FontSize', base_font_size, 'Interpreter','LaTex', 'FontWeight', 'bold');
   grid on;
   hold off
   %load('Data/Model Disagreement/MatFiles100Kyears/ExcessDemandRiskExposureBaseModel.mat', 'DelVec', 'XEalfamean', 'XEalfavol', 'XEalfaskew', 'XEalfakurt');
   plotLErho = figure;
   hold on
   [ax2, hl2, hr2] = plotyy(DelVec, XEalfamean, DelVec, XEalfavol);
   set(hl2, 'Color', 'b', 'LineWidth', 2, 'LineStyle', '--');
   set(hr2, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '-.');
   xlabel('Disagreement ($\Delta$)', 'FontSize', base_font_size, 'Interpreter','LaTex'); 
   set(get(ax2(1),'Ylabel'),'String','Unconditional Mean', 'FontSize', base_font_size);
   set(get(ax2(2),'Ylabel'),'String','Unconditional Volatility', 'FontSize', base_font_size);
   title('Demand Risk Exposure', 'FontSize', base_font_size, 'Interpreter','LaTex', 'FontWeight', 'bold');
   grid on;
   hold off
   
   plotXEalfa = figure;
   hold on
   [ax2, hl2, hr2] = plotyy(DelVec, XEalfamean, DelVec, XEalfavol);
   set(hl2, 'Color', 'b', 'LineWidth', 2, 'LineStyle', '--');
   set(hr2, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '-.');
   xlabel('Disagreement ($\Delta$)', 'FontSize', base_font_size, 'Interpreter','LaTex'); 
   set(get(ax2(1),'Ylabel'),'String','Unconditional Mean', 'FontSize', base_font_size);
   set(get(ax2(2),'Ylabel'),'String','Unconditional Volatility', 'FontSize', base_font_size);
   title('Demand Risk Exposure', 'FontSize', base_font_size, 'Interpreter','LaTex', 'FontWeight', 'bold');
   grid on;
   hold off
   %load('Data/Model Disagreement/MatFiles100Kyears/TradingVolumeBaseModel.mat', 'DelVec', 'TVolmean', 'TVolvol', 'TVolskew', 'TVolkurt');
   plotTVol = figure;
   hold on
   [ax2, hl2, hr2] = plotyy(DelVec, TVolmean, DelVec, TVolvol);
   set(hl2, 'Color', 'b', 'LineWidth', 2, 'LineStyle', '--');
   set(hr2, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '-.');
   xlabel('Disagreement ($\Delta$)', 'FontSize', base_font_size, 'Interpreter','LaTex'); 
   set(get(ax2(1),'Ylabel'),'String','Unconditional Mean', 'FontSize', base_font_size);
   set(get(ax2(2),'Ylabel'),'String','Unconditional Volatility', 'FontSize', base_font_size);
   title('Trading Volume', 'FontSize', base_font_size, 'Interpreter','LaTex', 'FontWeight', 'bold');
   grid on;
   hold off
   
end   
   

if sl == 8   
   
   %%
   load('LeverageandPredictability');
   %%
   %
        figure;
        dim = length(DelVec); ddim = 5;
        p1 = plot(DelVec, PDslopeMTH*12, '-.rx', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on;
        p2 = plot(DelVec, PDslopeYR1, '--bo', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p3 = plot(DelVec, PDslopeYR5, 'k', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p4 = plot(DelVec, PDslopeYR10, ':ms', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold off
        %
        xlabel('Disagreement ($\Delta$)', 'FontSize', base_font_size, 'Interpreter','LaTex'); 
        ylabel('Slope of PD-ratio (in bp)', 'FontSize', base_font_size );
        title('Stock Market Predictability', 'FontSize', base_font_size, 'Interpreter','LaTex', 'FontWeight', 'bold');
        legend( {'one-month', 'one-year', 'five-year', 'ten-year'}, 'Location', 'Best', 'FontSize', base_font_size);
        %set(gca,'XTick', [0; 0.1; 0.25; fCcovLE(2); 0.5; 0.65; 0.75; 0.9; 1]);            
        %set(gca,'XTickLabel',{'0', '0.1', '0.25', '0.35', '0.5', '0.65', '0.75', '0.9', '1'})   
        Yvec = [-25; -20; -15; -10; -5; 0]/10000;
        set(gca,'YTick', Yvec );            
        set(gca,'YTickLabel',{'-25', '-20', '-15', '-10', '-5', '0'});        
        %set(gca,'YTickLabel',{'-70', '-53', '-25', '-5', '0', '11', '25', '42', '50', '75', '100', '125', '135', '150'})        
        grid on
        %%
        %
        figure;
        dim = length(DelVec); ddim = 5;
        p1 = plot(DelVec, RsqMTH, '-.rx', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on;
        p2 = plot(DelVec, RsqYR1, '--bo', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p3 = plot(DelVec, RsqYR5, 'k', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p4 = plot(DelVec, RsqYR10, ':ms', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold off
        %
        xlabel('Disagreement $\left(\Delta \right)$', 'FontSize', base_font_size, 'Interpreter','LaTex'); 
        ylabel('Explanatory Power of PD-ratio ($R^2$)', 'Interpreter','LaTex', 'FontSize', base_font_size );
        title('Stock Market Predictability', 'FontSize', base_font_size, 'Interpreter','LaTex', 'FontWeight', 'bold');
        legend( {'one-month', 'one-year', 'five-year', 'ten-year'}, 'Location', 'Best', 'FontSize', base_font_size);
        %set(gca,'XTick', [0; 0.1; 0.25; fCcovLE(2); 0.5; 0.65; 0.75; 0.9; 1]);            
        %set(gca,'XTickLabel',{'0', '0.1', '0.25', '0.35', '0.5', '0.65', '0.75', '0.9', '1'})   
        Yvec = [0; 2.5; 5; 7.5; 10; 12.5; 15]/100;
        set(gca,'YTick', Yvec );            
        set(gca,'YTickLabel',{'0', '0.025', '0.05', '0.075', '0.10', '0.125', '0.15'});        
        %set(gca,'YTickLabel',{'-70', '-53', '-25', '-5', '0', '11', '25', '42', '50', '75', '100', '125', '135', '150'})        
        grid on
        %%
        %
        figure;
        dim = length(DelVec); ddim = 5;
        p1 = plot(DelVec, rhorxmthpd, '-.rx', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on;
        p2 = plot(DelVec, rhorx1pd, '--bo', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p3 = plot(DelVec, rhorx5pd, 'k', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p4 = plot(DelVec, rhorx10pd, ':ms', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold off
        %
        xlabel('Disagreement ($\Delta$)', 'FontSize', base_font_size, 'Interpreter','LaTex'); 
        ylabel('Correlation of PD-ratio and Equity Premium', 'FontSize', base_font_size );
        title('Stock Market Predictability', 'FontSize', base_font_size, 'Interpreter','LaTex', 'FontWeight', 'bold');
        legend( {'one-month', 'one-year', 'five-year', 'ten-year'}, 'Location', 'Best', 'FontSize', base_font_size);
        %set(gca,'XTick', [0; 0.1; 0.25; fCcovLE(2); 0.5; 0.65; 0.75; 0.9; 1]);            
        %set(gca,'XTickLabel',{'0', '0.1', '0.25', '0.35', '0.5', '0.65', '0.75', '0.9', '1'})   
        %Yvec = [-25; -20; -15; -10; -5; 0]/10000;
        %set(gca,'YTick', Yvec );            
        %set(gca,'YTickLabel',{'-25', '-20', '-15', '-10', '-5', '0'});        
        %set(gca,'YTickLabel',{'-70', '-53', '-25', '-5', '0', '11', '25', '42', '50', '75', '100', '125', '135', '150'})        
        grid on
        %%
   load('CorrelationPuzzleBaseModel.mat', 'DelVec', 'rhoPuzzle1', 'rhoPuzzle5', 'rhoPuzzle10');
        %
        %
        %
        %%
        figure;
        dim = length(DelVec); ddim = 5;
        p1 = plot(DelVec, rhoPuzzle1, '-.rx', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on;
        p2 = plot(DelVec, rhoPuzzle5, '--bo', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p3 = plot(DelVec, rhoPuzzle10, 'k', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        %hold on
        %p4 = plot(DelVec, rhoPuzzle10, ':ms', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold off
        %
        xlabel('Disagreement ($\Delta$)', 'FontSize', base_font_size, 'Interpreter','LaTex'); 
        ylabel('Return-Consumption Correlation', 'FontSize', base_font_size );
        title('Correlation Puzzle', 'FontSize', base_font_size, 'Interpreter','LaTex', 'FontWeight', 'bold');
        legend( {'one-year', 'five-year', 'ten-year'}, 'Location', 'Best', 'FontSize', base_font_size);
        %legend( {'one-month', 'one-year', 'five-year', 'ten-year'}, 'Location', 'Best', 'FontSize', base_font_size);
        %set(gca,'XTick', [0; 0.1; 0.25; fCcovLE(2); 0.5; 0.65; 0.75; 0.9; 1]);            
        %set(gca,'XTickLabel',{'0', '0.1', '0.25', '0.35', '0.5', '0.65', '0.75', '0.9', '1'})   
        %Yvec = [-25; -20; -15; -10; -5; 0]/10000;
        %set(gca,'YTick', Yvec );            
        %set(gca,'YTickLabel',{'-25', '-20', '-15', '-10', '-5', '0'});        
        %set(gca,'YTickLabel',{'-70', '-53', '-25', '-5', '0', '11', '25', '42', '50', '75', '100', '125', '135', '150'})        
        grid on
        %figure;
        %plot(DelVec, rhoPuzzlemth, 'm', DelVec, rhoPuzzle1, 'r', DelVec, rhoPuzzle5, 'b', DelVec, rhoPuzzle10, 'k');     
        
     %load('LeverageEffectBaseModel.mat');
     load('OnlineAppendixMatlabUnconditionalStatsBetter.mat', 'DelVec', 'rhoLEdt');
       %%
        figure;
        plot(DelVec, rhoLEdt, 'k', 'LineWidth', 2);
        xlabel('Disagreement ($\Delta$)', 'FontSize', base_font_size, 'Interpreter','LaTex'); 
        ylabel('Return-Volatility Correlation ($\mathrm{E}[\tilde{\rho}_{LE}]$)', 'FontSize', base_font_size, 'Interpreter','LaTex'); 
        title('Black''s "Leverage" Effect', 'FontSize', base_font_size, 'FontWeight', 'bold');
        grid on
        %     
       %
         
end

if sl == 8.2   
   
   %%
   load('LeverageandPredictability');
   %%
   %
        figure;
        dim = length(DelVec); ddim = 5;
        p1 = plot(DelVec, rhoR0PD, '-.rx', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on;
        p2 = plot(DelVec, rhorx1pd, '--bo', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p3 = plot(DelVec, rhorx5pd, 'k', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p4 = plot(DelVec, rhorx10pd, ':ms', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        %hold on
        %p4 = plot([fCcovLE(2), fCcovLE(2)], [ymin, 0], ':k', [fmin, fCcovLE(2)], [0, 0], ':k', 'LineWidth', 2);
        %hold on
        %p5 = plot([fCcovLE(3), fCcovLE(3)], [ymin, 0], ':k', [fmin, fCcovLE(3)], [0, 0], ':k', 'LineWidth', 2);
        hold off
        %
        xlabel('Disagreement ($\Delta$)', 'FontSize', base_font_size, 'Interpreter','LaTex'); 
        ylabel('Correlation of Returns and PD-ratio', 'FontSize', base_font_size );
        title('Stock Market Predictaibility', 'FontSize', base_font_size, 'Interpreter','LaTex', 'FontWeight', 'bold');
        legend( 'local', 'one-year', 'five-year', 'ten-year', 'Location', 'Best');
        %set(gca,'XTick', [0; 0.1; 0.25; fCcovLE(2); 0.5; 0.65; 0.75; 0.9; 1]);            
        %set(gca,'XTickLabel',{'0', '0.1', '0.25', '0.35', '0.5', '0.65', '0.75', '0.9', '1'})        
        %set(gca,'YTick', [-0.007; fClamMinVal(3); -0.0025; fClamMinVal(2); 0; lamM(1,1); 0.0025; fClamMaxVal(2);  0.005;  0.0075; 0.01; 0.0125; fClamMaxVal(3);  0.015]);            
        %set(gca,'YTickLabel',{'-0.70', '-0.53', '-0.25', '-0.05', '0', '0.11', '0.25', '0.42', '0.50', '0.75', '1.00', '1.25', '1.35', '1.5'})        
        %set(gca,'YTickLabel',{'-70', '-53', '-25', '-5', '0', '11', '25', '42', '50', '75', '100', '125', '135', '150'})        
        grid on
        %
   load('Data/Model Disagreement/MatFiles100Kyears/LeverageEffectBaseModel.mat', 'DelVec', 'rhor0sig', 'rhormthsig', 'rhor1sig');
   %
        %
        figure;
        ddim = 5;
        p1 = plot(DelVec, rhor0sig, '-.rx', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on;
        p2 = plot(DelVec, rhormthsig, '--bo', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p3 = plot(DelVec, rhor1sig, 'k', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        %hold on
        %p4 = plot([fCcovLE(2), fCcovLE(2)], [ymin, 0], ':k', [fmin, fCcovLE(2)], [0, 0], ':k', 'LineWidth', 2);
        %hold on
        %p5 = plot([fCcovLE(3), fCcovLE(3)], [ymin, 0], ':k', [fmin, fCcovLE(3)], [0, 0], ':k', 'LineWidth', 2);
        hold off
        %
        xlabel('Disagreement ($\Delta$)', 'FontSize', base_font_size, 'Interpreter','LaTex'); 
        ylabel('Correlation of Returns and Future Return volatility', 'FontSize', base_font_size );
        title('Leverage Effect', 'FontSize', base_font_size, 'Interpreter','LaTex', 'FontWeight', 'bold');
        legend( 'local', 'one-mth', 'one-year', 'Location', 'Best');
        %set(gca,'XTick', [0; 0.1; 0.25; fCcovLE(2); 0.5; 0.65; 0.75; 0.9; 1]);            
        %set(gca,'XTickLabel',{'0', '0.1', '0.25', '0.35', '0.5', '0.65', '0.75', '0.9', '1'})        
        %set(gca,'YTick', [-0.007; fClamMinVal(3); -0.0025; fClamMinVal(2); 0; lamM(1,1); 0.0025; fClamMaxVal(2);  0.005;  0.0075; 0.01; 0.0125; fClamMaxVal(3);  0.015]);            
        %set(gca,'YTickLabel',{'-0.70', '-0.53', '-0.25', '-0.05', '0', '0.11', '0.25', '0.42', '0.50', '0.75', '1.00', '1.25', '1.35', '1.5'})        
        %set(gca,'YTickLabel',{'-70', '-53', '-25', '-5', '0', '11', '25', '42', '50', '75', '100', '125', '135', '150'})        
        grid on
        
   
   load('Data/Model Disagreement/MatFiles100Kyears/CorrelationPuzzleBaseModel.mat', 'DelVec', 'rhoPuzzle1', 'rhoPuzzle5', 'rhoPuzzle10');
            
       %
        figure;
        ddim = 5;
        p1 = plot(DelVec, rhoPuzzle1, '-.rx', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on;
        p2 = plot(DelVec, rhoPuzzle5, '--bo', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p3 = plot(DelVec, rhoPuzzle10, 'k', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        %hold on
        %p4 = plot([fCcovLE(2), fCcovLE(2)], [ymin, 0], ':k', [fmin, fCcovLE(2)], [0, 0], ':k', 'LineWidth', 2);
        %hold on
        %p5 = plot([fCcovLE(3), fCcovLE(3)], [ymin, 0], ':k', [fmin, fCcovLE(3)], [0, 0], ':k', 'LineWidth', 2);
        hold off
        %
        xlabel('Disagreement ($\Delta$)', 'FontSize', base_font_size, 'Interpreter','LaTex'); 
        ylabel('Correlation of Returns and Output growth', 'FontSize', base_font_size );
        title('Correlation Puzzle', 'FontSize', base_font_size, 'Interpreter','LaTex', 'FontWeight', 'bold');
        legend( 'one-year', 'five-year', 'ten-year', 'Location', 'Best');
        %set(gca,'XTick', [0; 0.1; 0.25; fCcovLE(2); 0.5; 0.65; 0.75; 0.9; 1]);            
        %set(gca,'XTickLabel',{'0', '0.1', '0.25', '0.35', '0.5', '0.65', '0.75', '0.9', '1'})        
        %set(gca,'YTick', [-0.007; fClamMinVal(3); -0.0025; fClamMinVal(2); 0; lamM(1,1); 0.0025; fClamMaxVal(2);  0.005;  0.0075; 0.01; 0.0125; fClamMaxVal(3);  0.015]);            
        %set(gca,'YTickLabel',{'-0.70', '-0.53', '-0.25', '-0.05', '0', '0.11', '0.25', '0.42', '0.50', '0.75', '1.00', '1.25', '1.35', '1.5'})        
        %set(gca,'YTickLabel',{'-70', '-53', '-25', '-5', '0', '11', '25', '42', '50', '75', '100', '125', '135', '150'})        
        grid on
        %%
        %     
end

if sl == 8.3   
   % 
   % OLD AND NOT CORRECT
   %
   load('Data/Model Disagreement/MatFiles100Kyears/PredictabilityBaseModel.mat', 'DelVec', 'rhoR0PD', 'rhoR1PD', 'rhoR5PD', 'rhoR10PD', 'rhorx1pd', 'rhorx5pd', 'rhorx10pd');
        %
        figure;
        dim = length(DelVec); ddim = 5;
        p1 = plot(DelVec, rhoR0PD, '-.rx', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on;
        p2 = plot(DelVec, rhorx1pd, '--bo', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p3 = plot(DelVec, rhorx5pd, 'k', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p4 = plot(DelVec, rhorx10pd, ':ms', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        %hold on
        %p4 = plot([fCcovLE(2), fCcovLE(2)], [ymin, 0], ':k', [fmin, fCcovLE(2)], [0, 0], ':k', 'LineWidth', 2);
        %hold on
        %p5 = plot([fCcovLE(3), fCcovLE(3)], [ymin, 0], ':k', [fmin, fCcovLE(3)], [0, 0], ':k', 'LineWidth', 2);
        hold off
        %
        xlabel('Disagreement ($\Delta$)', 'FontSize', base_font_size, 'Interpreter','LaTex'); 
        ylabel('Correlation of Returns and PD-ratio', 'FontSize', base_font_size );
        title('Stock Market Predictaibility', 'FontSize', base_font_size, 'Interpreter','LaTex', 'FontWeight', 'bold');
        legend( 'local', 'one-year', 'five-year', 'ten-year', 'Location', 'Best');
        %set(gca,'XTick', [0; 0.1; 0.25; fCcovLE(2); 0.5; 0.65; 0.75; 0.9; 1]);            
        %set(gca,'XTickLabel',{'0', '0.1', '0.25', '0.35', '0.5', '0.65', '0.75', '0.9', '1'})        
        %set(gca,'YTick', [-0.007; fClamMinVal(3); -0.0025; fClamMinVal(2); 0; lamM(1,1); 0.0025; fClamMaxVal(2);  0.005;  0.0075; 0.01; 0.0125; fClamMaxVal(3);  0.015]);            
        %set(gca,'YTickLabel',{'-0.70', '-0.53', '-0.25', '-0.05', '0', '0.11', '0.25', '0.42', '0.50', '0.75', '1.00', '1.25', '1.35', '1.5'})        
        %set(gca,'YTickLabel',{'-70', '-53', '-25', '-5', '0', '11', '25', '42', '50', '75', '100', '125', '135', '150'})        
        grid on
        %
   load('Data/Model Disagreement/MatFiles100Kyears/LeverageEffectBaseModel.mat', 'DelVec', 'rhor0sig', 'rhormthsig', 'rhor1sig');
   %
        %
        figure;
        ddim = 5;
        p1 = plot(DelVec, rhor0sig, '-.rx', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on;
        p2 = plot(DelVec, rhormthsig, '--bo', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p3 = plot(DelVec, rhor1sig, 'k', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        %hold on
        %p4 = plot([fCcovLE(2), fCcovLE(2)], [ymin, 0], ':k', [fmin, fCcovLE(2)], [0, 0], ':k', 'LineWidth', 2);
        %hold on
        %p5 = plot([fCcovLE(3), fCcovLE(3)], [ymin, 0], ':k', [fmin, fCcovLE(3)], [0, 0], ':k', 'LineWidth', 2);
        hold off
        %
        xlabel('Disagreement ($\Delta$)', 'FontSize', base_font_size, 'Interpreter','LaTex'); 
        ylabel('Correlation of Returns and Future Return volatility', 'FontSize', base_font_size );
        title('Leverage Effect', 'FontSize', base_font_size, 'Interpreter','LaTex', 'FontWeight', 'bold');
        legend( 'local', 'one-mth', 'one-year', 'Location', 'Best');
        %set(gca,'XTick', [0; 0.1; 0.25; fCcovLE(2); 0.5; 0.65; 0.75; 0.9; 1]);            
        %set(gca,'XTickLabel',{'0', '0.1', '0.25', '0.35', '0.5', '0.65', '0.75', '0.9', '1'})        
        %set(gca,'YTick', [-0.007; fClamMinVal(3); -0.0025; fClamMinVal(2); 0; lamM(1,1); 0.0025; fClamMaxVal(2);  0.005;  0.0075; 0.01; 0.0125; fClamMaxVal(3);  0.015]);            
        %set(gca,'YTickLabel',{'-0.70', '-0.53', '-0.25', '-0.05', '0', '0.11', '0.25', '0.42', '0.50', '0.75', '1.00', '1.25', '1.35', '1.5'})        
        %set(gca,'YTickLabel',{'-70', '-53', '-25', '-5', '0', '11', '25', '42', '50', '75', '100', '125', '135', '150'})        
        grid on
        
   
   load('Data/Model Disagreement/MatFiles100Kyears/CorrelationPuzzleBaseModel.mat', 'DelVec', 'rhoPuzzle1', 'rhoPuzzle5', 'rhoPuzzle10');
            
       %
        figure;
        ddim = 5;
        p1 = plot(DelVec, rhoPuzzle1, '-.rx', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on;
        p2 = plot(DelVec, rhoPuzzle5, '--bo', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        hold on
        p3 = plot(DelVec, rhoPuzzle10, 'k', 'MarkerIndices', 1:ddim:dim, 'LineWidth', 2);
        %hold on
        %p4 = plot([fCcovLE(2), fCcovLE(2)], [ymin, 0], ':k', [fmin, fCcovLE(2)], [0, 0], ':k', 'LineWidth', 2);
        %hold on
        %p5 = plot([fCcovLE(3), fCcovLE(3)], [ymin, 0], ':k', [fmin, fCcovLE(3)], [0, 0], ':k', 'LineWidth', 2);
        hold off
        %
        xlabel('Disagreement ($\Delta$)', 'FontSize', base_font_size, 'Interpreter','LaTex'); 
        ylabel('Correlation of Returns and Output growth', 'FontSize', base_font_size );
        title('Correlation Puzzle', 'FontSize', base_font_size, 'Interpreter','LaTex', 'FontWeight', 'bold');
        legend( 'one-year', 'five-year', 'ten-year', 'Location', 'Best');
        %set(gca,'XTick', [0; 0.1; 0.25; fCcovLE(2); 0.5; 0.65; 0.75; 0.9; 1]);            
        %set(gca,'XTickLabel',{'0', '0.1', '0.25', '0.35', '0.5', '0.65', '0.75', '0.9', '1'})        
        %set(gca,'YTick', [-0.007; fClamMinVal(3); -0.0025; fClamMinVal(2); 0; lamM(1,1); 0.0025; fClamMaxVal(2);  0.005;  0.0075; 0.01; 0.0125; fClamMaxVal(3);  0.015]);            
        %set(gca,'YTickLabel',{'-0.70', '-0.53', '-0.25', '-0.05', '0', '0.11', '0.25', '0.42', '0.50', '0.75', '1.00', '1.25', '1.35', '1.5'})        
        %set(gca,'YTickLabel',{'-70', '-53', '-25', '-5', '0', '11', '25', '42', '50', '75', '100', '125', '135', '150'})        
        grid on
        %%
        %
   
        %         
end                     
        % -----------------------------------------------------------------------------------------------
if sl == 9           % Unconditional distribution --- histograms
        % -----------------------------------------------------------------------------------------------
        
        load('Data/Model Disagreement/ConsumptionShareHist.mat');
        
        Mbars = 75;
        figure;
        h1=histogram(fVecBaseDisH);
        set(h1,'FaceColor','r','EdgeColor','r','facealpha',0.8)
        hold on
        h2=histogram(fVecBaseDisM);
        set(h2,'FaceColor','g','EdgeColor','g','facealpha',0.5);
        hold on
        h3=histogram(fVecBaseDis0);
        set(h3,'FaceColor','b','EdgeColor','b','facealpha',0.3);
        ylabel('Frequency in millions', 'FontSize', base_font_size);
        xlabel('Consumption share, f', 'FontSize', base_font_size);
        hold on;
        line([mean(fVecBaseDisH), mean(fVecBaseDisH)], ylim, 'LineWidth', 2, 'Color', 'r');
        hold on;
        line([mean(fVecBaseDisM), mean(fVecBaseDisM)], ylim, 'LineWidth', 2, 'Color', 'g');
        hold on;
        line([mean(fVecBaseDis0), mean(fVecBaseDis0)], ylim, 'LineWidth', 2, 'Color', 'b');
        legend('High Dis (\Delta = 0.8)','Med Dis (\Delta = 0.4)','No Dis (\Delta = 0)','Mean High Dis','Mean Med Dis','Mean No Dis');
        xlim([0 1]);
        hold off;
        set(gca,'YTick', [ 0; 0.5; 1; 1.5; 2; 2.5; 3; 3.5 ]*10^5);            
        set(gca,'YTickLabel',{'0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35'})   
        
end
        % -----------------------------------------------------------------------------------------------
if sl == 9.1      % The uncondtional distribution of the consumption share for different disagreement    
       % -----------------------------------------------------------------------------------------------
    %
    T = 1000000;
    %
    if T == 1000000
        sv = 1;
    else
        sv = 0;
    end
    %
    if sv == 1
        tic;
        %
        if T == 100
            load('Data/Model Disagreement/MatFiles100years/BMandAlfa.mat', 'dZvec', 'T', 'dt', 'N', 'kap', 'lbar', 'sigl', 'alfa0', 'alfaVec', 'lVec', 'etime' );
        elseif T == 1000
            load('Data/Model Disagreement/MatFiles1Kyears/BMandAlfa.mat', 'dZvec', 'T', 'dt', 'N', 'kap', 'lbar', 'sigl', 'alfa0', 'alfaVec', 'lVec', 'etime' );
        elseif T == 10000
            load('Data/Model Disagreement/MatFiles10Kyears/BMandAlfa.mat', 'dZvec', 'T', 'dt', 'N', 'kap', 'lbar', 'sigl', 'alfa0', 'alfaVec', 'lVec', 'etime' );
        elseif T == 100000
            load('Data/Model Disagreement/MatFiles100Kyears/BMandAlfa.mat', 'dZvec', 'T', 'dt', 'N', 'kap', 'lbar', 'sigl', 'alfa0', 'alfaVec', 'lVec', 'etime' );
        elseif T == 1000000
            load('Data/Model Disagreement/MatFiles1Myears/BMandAlfa.mat', 'dZvec', 'T', 'dt', 'N', 'kap', 'lbar', 'sigl', 'alfa0', 'alfaVec', 'lVec', 'etime' );
        end
        %
        nu = 0.02;
        f0 = alfa0;
        %
        %DelVec = [ (linspace(0,0.4,5))'; (linspace(0.4,0.8, 5))'; (linspace(0.8,1, 5))'];
        %Del0 = DelVec(1);
        %DelM = DelVec(6);
        %DelH = DelVec(11);
        DelVec = [ (linspace(0,0.4,20))'; (linspace(0.4,0.8, 20))'; (linspace(0.8,1, 10))'];
        Del0 = DelVec(1);
        DelM = DelVec(21);
        DelH = DelVec(41);
        dVec = DelVec*sigl/(2*kap);
        dim2 = length(dVec);
        %
        rhoA = 0.001;
        rhoB = 0.05;
        rho = mean([rhoA;rhoB]);
        drhoBase = rhoB-rhoA;
        drhoLow = 0.01;
        drhoMed = 0.025;
        %
        fMatBase = -999*ones(N,dim2);
        fMatLow = -999*ones(N,dim2);
        fMatMed = -999*ones(N,dim2);
        %
        %        
        indxRho = 1;
        while indxRho <= 3   % three different Del rho
            if indxRho == 1
                parfor indx2 = 1:dim2
                %for indx2 = 1:dim2
                    d = dVec(indx2);
                    %
                    fMatBase(:,indx2) = OnlineAppendixSimulateConsumptionShareDogmaticLongRunMean(kap, lbar, sigl, d, nu, rho, drhoBase, dZvec, dt, alfa0, f0);
                    %
                end
                %
                fVecBaseDis0 = fMatBase(:,1);
                fVecBaseDisM = fMatBase(:,21);
                fVecBaseDisH = fMatBase(:,41);
                %
            elseif indxRho == 2
                parfor indx2 = 1:dim2
                %for indx2 = 1:dim2
                    d = dVec(indx2);
                    %
                    fMatMed(:,indx2) = OnlineAppendixSimulateConsumptionShareDogmaticLongRunMean(kap, lbar, sigl, d, nu, rho, drhoMed, dZvec, dt, alfa0, f0);
                end
                %
                fVecMedDis0 = fMatMed(:,1);
                fVecMedDisM = fMatMed(:,21);
                fVecMedDisH = fMatMed(:,41);
                %
            elseif indxRho == 3
                parfor indx2 = 1:dim2
                %for indx2 = 1:dim2
                    d = dVec(indx2);
                    %
                    fMatLow(:,indx2) = OnlineAppendixSimulateConsumptionShareDogmaticLongRunMean(kap, lbar, sigl, d, nu, rho, drhoLow, dZvec, dt, alfa0, f0);
                    %    
                end
                %
                fVecLowDis0 = fMatLow(:,1);
                fVecLowDisM = fMatLow(:,21);
                fVecLowDisH = fMatLow(:,41);
                %
            end
            indxRho = indxRho+1; 
        end
        %
        etimef = toc;
        %
        if T == 100
            save('Data/Model Disagreement/MatFiles100years/StateVariables2Dis.mat');
        elseif T == 1000
            save('Data/Model Disagreement/MatFiles1Kyears/StateVariables2Dis.mat');
        elseif T == 10000
            save('Data/Model Disagreement/MatFiles10Kyears/StateVariables2Dis.mat');
        elseif T == 100000
            save('Data/Model Disagreement/MatFiles100Kyears/StateVariables2Dis.mat');
        %
        end
    else
        %
        if T == 100
            load('Data/Model Disagreement/MatFiles100years/StateVariables2Dis.mat');
        elseif T == 1000
            load('Data/Model Disagreement/MatFiles1Kyears/StateVariables2Dis.mat');
        elseif T == 10000
            load('Data/Model Disagreement/MatFiles10Kyears/StateVariables2Dis.mat');
        elseif T == 100000
            load('Data/Model Disagreement/MatFiles100Kyears/StateVariables2Dis.mat');
        end
        %
    end
    %
    %%
    if T >= 100000
        Mbars = 75;
        figure;
        h1=histogram(fVecBaseDisH);
        set(h1,'FaceColor','r','EdgeColor','r','facealpha',0.8)
        hold on
        h2=histogram(fVecBaseDisM);
        set(h2,'FaceColor','g','EdgeColor','g','facealpha',0.5);
        hold on
        h3=histogram(fVecBaseDis0);
        set(h3,'FaceColor','b','EdgeColor','b','facealpha',0.3);
        ylabel('Frequency in millions', 'FontSize', base_font_size);
        xlabel('Consumption share, f', 'FontSize', base_font_size);
        hold on;
        line([mean(fVecBaseDisH), mean(fVecBaseDisH)], ylim, 'LineWidth', 2, 'Color', 'r');
        hold on;
        line([mean(fVecBaseDisM), mean(fVecBaseDisM)], ylim, 'LineWidth', 2, 'Color', 'g');
        hold on;
        line([mean(fVecBaseDis0), mean(fVecBaseDis0)], ylim, 'LineWidth', 2, 'Color', 'b');
        legend('High Dis (\Delta = 0.8)','Med Dis (\Delta = 0.4)','No Dis (\Delta = 0)','Mean High Dis','Mean Med Dis','Mean No Dis');
        xlim([0 1]);
        hold off;
        set(gca,'YTick', [ 0; 0.5; 1; 1.5; 2; 2.5; 3; 3.5 ]*10^5);            
        set(gca,'YTickLabel',{'0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35'})               
        
        %print('MatFiles100Kyears/fig/JFround1_HIST2DisBase_f','-dpdf')
        %%
        fprintf('Mean of f is %2.2f percent with no disagreement \n',mean(fVecBaseDis0)*100  )
        fprintf('Mean of f is %2.2f percent with med disagreement \n',mean(fVecBaseDisM)*100  )
        fprintf('Mean of f is %2.2f percent with high disagreement \n',mean(fVecBaseDisH)*100  )
        fprintf('vola of f is %2.2f percent with no disagreement \n',std(fVecBaseDis0)*100  )
        fprintf('vola of f is %2.2f percent with med disagreement \n',std(fVecBaseDisM)*100  )
        fprintf('vola of f is %2.2f percent with high disagreement \n',std(fVecBaseDisH)*100  )
        %%
        fmeanBase = mean(fMatBase);
        fmeanMed = mean(fMatMed);
        fmeanLow = mean(fMatLow);
        %fmeanH = mean
        % 
        figure;
        plot(DelVec, fmeanLow, 'r', DelVec, fmeanMed, 'b', DelVec, fmeanBase, 'k', 'LineWidth', 2);
        %legend( '$\Delta \rho = 1\%$', '$\Delta \rho = 2.5\%$', '$\Delta \rho = 5\%$', 'Location', 'Best', 'Interpreter','LaTex'); 
        xlabel('Disagreement ($\Delta$)', 'FontSize', 12, 'Interpreter','LaTex'); 
        ylabel('Consumption Share', 'FontSize', 12 );
        title([ 'NewBase Case ( $\rho$ = ',num2str(rho),' )'], 'FontSize', 16, 'Interpreter','LaTex', 'FontWeight', 'bold');
        hold off
        grid on
        %
    else
        figure;
        h1=histogram(fVecBaseDisH);
        set(h1,'FaceColor','r','EdgeColor','w','facealpha',0.8)
        hold on
        h2=histogram(fVecBaseDisM);
        set(h2,'FaceColor','g','EdgeColor','w','facealpha',0.5);
        hold on
        h3=histogram(fVecBaseDis0);
        set(h3,'FaceColor','b','EdgeColor','w','facealpha',0.3);
        ylabel('counts')
        xlabel('Consumption share, f')
        hold on;
        line([mean(fVecBaseDisH), mean(fVecBaseDisH)], ylim, 'LineWidth', 2, 'Color', 'r');
        hold on;
        line([mean(fVecBaseDisM), mean(fVecBaseDisM)], ylim, 'LineWidth', 2, 'Color', 'g');
        hold on;
        line([mean(fVecBaseDis0), mean(fVecBaseDis0)], ylim, 'LineWidth', 2, 'Color', 'b');
        legend('High Dis','Med Dis','No Dis','Mean High Dis','Mean Med Dis','Mean No Dis')
        %axis([0 1 0 0.9*T]);
        hold off;
        %
    end
end

       % -----------------------------------------------------------------------------------------------    
if sl == 9.2      % The uncondtional distribution of the PD ratio for different disagreement    
       % -----------------------------------------------------------------------------------------------
    %
    T = 1000000;
    %
    if T == 1000000
        sv = 1;
    else
        sv = 0;
    end
    %
    if sv == 1
        tic;
        %
        if T == 100
            load('Data/Model Disagreement/MatFiles100years/BMandAlfa.mat', 'dZvec', 'T', 'dt', 'N', 'kap', 'lbar', 'sigl', 'alfa0', 'alfaVec', 'lVec', 'etime' );
        elseif T == 1000
            load('Data/Model Disagreement/MatFiles1Kyears/BMandAlfa.mat', 'dZvec', 'T', 'dt', 'N', 'kap', 'lbar', 'sigl', 'alfa0', 'alfaVec', 'lVec', 'etime' );
        elseif T == 10000
            load('Data/Model Disagreement/MatFiles10Kyears/BMandAlfa.mat', 'dZvec', 'T', 'dt', 'N', 'kap', 'lbar', 'sigl', 'alfa0', 'alfaVec', 'lVec', 'etime' );
        elseif T == 100000
            load('Data/Model Disagreement/MatFiles100Kyears/BMandAlfa.mat', 'dZvec', 'T', 'dt', 'N', 'kap', 'lbar', 'sigl', 'alfa0', 'alfaVec', 'lVec', 'etime' );
        elseif T == 1000000
            load('Data/Model Disagreement/MatFiles1Myears/BMandAlfa.mat', 'dZvec', 'T', 'dt', 'N', 'kap', 'lbar', 'sigl', 'alfa0', 'alfaVec', 'lVec', 'etime' );
        end
        %
        nu = 0.02;
        f0 = alfa0;
        %
        %DelVec = [ (linspace(0,0.4,5))'; (linspace(0.4,0.8, 5))'; (linspace(0.8,1, 5))'];
        %Del0 = DelVec(1);
        %DelM = DelVec(6);
        %DelH = DelVec(11);
        DelVec = [ (linspace(0,0.4,20))'; (linspace(0.4,0.8, 20))'; (linspace(0.8,1, 10))'];
        Del0 = DelVec(1);
        DelM = DelVec(21);
        DelH = DelVec(41);
        dVec = DelVec*sigl/(2*kap);
        dim2 = length(dVec);
        %
        rhoA = 0.001;
        rhoB = 0.05;
        rho = mean([rhoA;rhoB]);
        drhoBase = rhoB-rhoA;
        drhoLow = 0.01;
        drhoMed = 0.025;
        %
        fMatBase = -999*ones(N,dim2);
        fMatLow = -999*ones(N,dim2);
        fMatMed = -999*ones(N,dim2);
        %
        %        
        indxRho = 1;
        while indxRho <= 3   % three different Del rho
            if indxRho == 1
                parfor indx2 = 1:dim2
                %for indx2 = 1:dim2
                    d = dVec(indx2);
                    %
                    fMatBase(:,indx2) = OnlineAppendixSimulateConsumptionShareDogmaticLongRunMean(kap, lbar, sigl, d, nu, rho, drhoBase, dZvec, dt, alfa0, f0);
                    %
                end
                %
                fVecBaseDis0 = fMatBase(:,1);
                fVecBaseDisM = fMatBase(:,21);
                fVecBaseDisH = fMatBase(:,41);
                %
            elseif indxRho == 2
                parfor indx2 = 1:dim2
                %for indx2 = 1:dim2
                    d = dVec(indx2);
                    %
                    fMatMed(:,indx2) = OnlineAppendixSimulateConsumptionShareDogmaticLongRunMean(kap, lbar, sigl, d, nu, rho, drhoMed, dZvec, dt, alfa0, f0);
                end
                %
                fVecMedDis0 = fMatMed(:,1);
                fVecMedDisM = fMatMed(:,21);
                fVecMedDisH = fMatMed(:,41);
                %
            elseif indxRho == 3
                parfor indx2 = 1:dim2
                %for indx2 = 1:dim2
                    d = dVec(indx2);
                    %
                    fMatLow(:,indx2) = OnlineAppendixSimulateConsumptionShareDogmaticLongRunMean(kap, lbar, sigl, d, nu, rho, drhoLow, dZvec, dt, alfa0, f0);
                    %    
                end
                %
                fVecLowDis0 = fMatLow(:,1);
                fVecLowDisM = fMatLow(:,21);
                fVecLowDisH = fMatLow(:,41);
                %
            end
            indxRho = indxRho+1; 
        end
        %
        etimef = toc;
        %
        if T == 100
            save('Data/Model Disagreement/MatFiles100years/StateVariables2Dis.mat');
        elseif T == 1000
            save('Data/Model Disagreement/MatFiles1Kyears/StateVariables2Dis.mat');
        elseif T == 10000
            save('Data/Model Disagreement/MatFiles10Kyears/StateVariables2Dis.mat');
        elseif T == 100000
            save('Data/Model Disagreement/MatFiles100Kyears/StateVariables2Dis.mat');
        %
        end
    else
        %
        if T == 100
            load('Data/Model Disagreement/MatFiles100years/StateVariables2Dis.mat');
        elseif T == 1000
            load('Data/Model Disagreement/MatFiles1Kyears/StateVariables2Dis.mat');
        elseif T == 10000
            load('Data/Model Disagreement/MatFiles10Kyears/StateVariables2Dis.mat');
        elseif T == 100000
            load('Data/Model Disagreement/MatFiles100Kyears/StateVariables2Dis.mat');
        end
        %
    end
    %
    rhoAMed = rho-drhoMed/2;
    rhoBMed = rho+drhoMed/2;
    rhoALow = rho-drhoLow/2;
    rhoBLow = rho+drhoLow/2;
        
    phiA  = 1/(rhoA+nu);
    phiB  = 1/(rhoB+nu);
    phiAMed = 1/(rhoAMed+nu);
    phiBMed = 1/(rhoBMed+nu);
    phiALow = 1/(rhoALow+nu);
    phiBLow = 1/(rhoBLow+nu);
    %
    PDvecBaseDisH = phiA*fVecBaseDisH + phiB*(1-fVecBaseDisH);
    PDvecBaseDisM = phiA*fVecBaseDisM + phiB*(1-fVecBaseDisM);
    PDvecBaseDis0 = phiA*fVecBaseDis0 + phiB*(1-fVecBaseDis0);
    %%
    % figures
    %
    save('PDhist', 'PDvecBaseDis0', 'PDvecBaseDisM', 'PDvecBaseDisH');
    
        Mbars = 75;
        figure;
        h1=histogram(PDvecBaseDisH);
        set(h1,'FaceColor','r','EdgeColor','w','facealpha',0.8)
        hold on
        h2=histogram(PDvecBaseDisM);
        set(h2,'FaceColor','g','EdgeColor','w','facealpha',0.5);
        hold on
        h3=histogram(PDvecBaseDis0);
        set(h3,'FaceColor','b','EdgeColor','w','facealpha',0.3);
        ylabel('counts')
        xlabel('PD-ratio')
        hold on;
        line([mean(PDvecBaseDisH), mean(PDvecBaseDisH)], ylim, 'LineWidth', 2, 'Color', 'r');
        hold on;
        line([mean(PDvecBaseDisM), mean(PDvecBaseDisM)], ylim, 'LineWidth', 2, 'Color', 'g');
        hold on;
        line([mean(PDvecBaseDis0), mean(PDvecBaseDis0)], ylim, 'LineWidth', 2, 'Color', 'b');
        legend('High Dis','Med Dis','No Dis','Mean High Dis','Mean Med Dis','Mean No Dis')
        %axis([0 1 0 0.9*T]);
        hold off;
        %print('MatFiles100Kyears/fig/JFround1_HIST2DisBase_f','-dpdf')
        %%
        fprintf('Mean of PD is %2.2f with no disagreement \n',mean(PDvecBaseDis0) )
        fprintf('Mean of PD is %2.2f with med disagreement \n',mean(PDvecBaseDisM) )
        fprintf('Mean of PD is %2.2f with high disagreement \n',mean(PDvecBaseDisH) )
        fprintf('vola of PD is %2.2f with no disagreement \n',std(PDvecBaseDis0) )
        fprintf('vola of PD is %2.2f with med disagreement \n',std(PDvecBaseDisM) )
        fprintf('vola of PD is %2.2f with high disagreement \n',std(PDvecBaseDisH)   )
        fprintf('vola of log PD is %2.2f with no disagreement \n',std(log(PDvecBaseDis0)) )
        fprintf('vola of log PD is %2.2f with med disagreement \n',std(log(PDvecBaseDisM)) )
        fprintf('vola of log PD is %2.2f with high disagreement \n',std(log(PDvecBaseDisH))   )
        %%
        PDmeanBase = phiA*mean(fMatBase) + phiB*mean(1-fMatBase);
        PDmeanMed = phiAMed*mean(fMatMed) + phiBMed*mean(1-fMatMed);
        PDmeanLow = phiALow*mean(fMatLow) + phiBLow*mean(1-fMatLow);
        %fmeanH = mean
        %%
        figure;
        plot(DelVec, PDmeanLow, 'r', DelVec, PDmeanMed, 'b', DelVec, PDmeanBase, 'k', 'LineWidth', 2);
        legend( '$\Delta \rho$ = 1 %', '$\Delta \rho$ = 2.5 %', '$\Delta \rho$ = 5%$', 'Location', 'Best', 'Interpreter','LaTex'); 
        xlabel('Disagreement ($\Delta$)', 'FontSize', 12, 'Interpreter','LaTex'); 
        ylabel('PD ratio', 'FontSize', 12 );
        title([ 'NewBase Case ( $\rho$ = ',num2str(rho),' )'], 'FontSize', 16, 'Interpreter','LaTex', 'FontWeight', 'bold');
        hold off
        grid on
        %
        PDBase = phiA*fMatBase + phiB*(1-fMatBase);
        PDMed = phiAMed*fMatMed + phiBMed*(1-fMatMed);
        PDLow = phiALow*fMatLow + phiBLow*(1-fMatLow);
        %
        %
        PDsigBase = std(log(PDBase));
        PDsigMed = std(log(PDMed));
        PDsigLow = std(log(PDLow));
        
        %fmeanH = mean
        %%
        figure;
        plot(DelVec, PDsigLow, 'r', DelVec, PDsigMed, 'b', DelVec, PDsigBase, 'k', 'LineWidth', 2);
        legend( '$\Delta$ $\rho$ = 1 %', '$\Delta \rho = 2.5\%$', '$\Delta \rho = 5\%$', 'Location', 'Best', 'Interpreter','LaTex'); 
        xlabel('Disagreement ($\Delta$)', 'FontSize', 12, 'Interpreter','LaTex'); 
        ylabel('volatility of log PD ratio', 'FontSize', 12 );
        title([ 'NewBase Case ( $\rho$ = ',num2str(rho),' )'], 'FontSize', 16, 'Interpreter','LaTex', 'FontWeight', 'bold');
        hold off
        grid on
        %%
        PDskewBase = skewness(log(PDBase));
        PDskewMed = skewness(log(PDMed));
        PDskewLow = skewness(log(PDLow));
        
        %fmeanH = mean
        %%
        figure;
        plot(DelVec, PDskewLow, 'r', DelVec, PDskewMed, 'b', DelVec, PDskewBase, 'k', 'LineWidth', 2);
        legend( '$\Delta$ $\rho$ = 1 %', '$\Delta \rho = 2.5\%$', '$\Delta \rho = 5\%$', 'Location', 'Best', 'Interpreter','LaTex'); 
        xlabel('Disagreement ($\Delta$)', 'FontSize', 12, 'Interpreter','LaTex'); 
        ylabel('skewness of log PD ratio', 'FontSize', 12 );
        title([ 'NewBase Case ( $\rho$ = ',num2str(rho),' )'], 'FontSize', 16, 'Interpreter','LaTex', 'FontWeight', 'bold');
        hold off
        grid on
        %%
        
        %%
        PDkurtBase = kurtosis(log(PDBase));
        PDkurtMed = kurtosis(log(PDMed));
        PDkurtLow = kurtosis(log(PDLow));
        
        %fmeanH = mean
        %%
        figure;
        plot(DelVec, PDkurtLow, 'r', DelVec, PDkurtMed, 'b', DelVec, PDkurtBase, 'k', 'LineWidth', 2);
        legend( '$\Delta$ $\rho$ = 1 %', '$\Delta \rho = 2.5\%$', '$\Delta \rho = 5\%$', 'Location', 'Best', 'Interpreter','LaTex'); 
        xlabel('Disagreement ($\Delta$)', 'FontSize', 12, 'Interpreter','LaTex'); 
        ylabel('kurtosis of log PD ratio', 'FontSize', 12 );
        title([ 'NewBase Case ( $\rho$ = ',num2str(rho),' )'], 'FontSize', 16, 'Interpreter','LaTex', 'FontWeight', 'bold');
        hold off
        grid on
        %%
        
        
        Mbars = 75;
        figure;
        h1=histogram(log(PDvecBaseDisH));
        set(h1,'FaceColor','r','EdgeColor','w','facealpha',0.8)
        hold on
        h2=histogram(log(PDvecBaseDisM));
        set(h2,'FaceColor','g','EdgeColor','w','facealpha',0.5);
        hold on
        h3=histogram(log(PDvecBaseDis0));
        set(h3,'FaceColor','b','EdgeColor','w','facealpha',0.3);
        ylabel('counts')
        xlabel('PD-ratio')
        hold on;
        line([mean(log(PDvecBaseDisH)), mean(log(PDvecBaseDisH))], ylim, 'LineWidth', 2, 'Color', 'r');
        hold on;
        line([mean(log(PDvecBaseDisM)), mean(log(PDvecBaseDisM))], ylim, 'LineWidth', 2, 'Color', 'g');
        hold on;
        line([mean(log(PDvecBaseDis0)), mean(log(PDvecBaseDis0))], ylim, 'LineWidth', 2, 'Color', 'b');
        legend('High Dis','Med Dis','No Dis','Mean High Dis','Mean Med Dis','Mean No Dis')
        %axis([0 1 0 0.9*T]);
        hold off;
        %print('MatFiles100Kyears/fig/JFround1_HIST2DisBase_f','-dpdf')
        %%
        fprintf('Mean of f is %2.2f with no disagreement \n',mean(log(PDvecBaseDis0) ))
        fprintf('Mean of f is %2.2f with med disagreement \n',mean(log(PDvecBaseDisM) ))
        fprintf('Mean of f is %2.2f with high disagreement \n',mean(log(PDvecBaseDisH) ))
        fprintf('vola of f is %2.2f with no disagreement \n',std(log(PDvecBaseDis0) ))
        fprintf('vola of f is %2.2f with med disagreement \n',std(log(PDvecBaseDisM) ))
        fprintf('vola of f is %2.2f with high disagreement \n',std(log(PDvecBaseDisH))   )
        %
        %
        %
        PDmat = log(PDBase);
        muVec = mean(PDmat)';
        sigVec = std(PDmat)';
        CritPos2sig = muVec+2*sigVec;
        CritPos1sig = muVec+sigVec;
        CritMean = muVec;
        CritNeg1sig = muVec-1*sigVec;
        CritNeg2sig = muVec-2*sigVec;
        %%
        check = max(isnan(PDmat));
        for indx = 1:length(muVec)
            ProbPos2sig(indx) = sum(PDmat(indx)>CritPos2sig(indx))/length(PDmat);
            ProbPos1sig(indx) = sum(PDmat(indx)>CritPos1sig(indx))/length(PDmat);
            ProbMean(indx) = sum(PDmat(indx)>CritMean(indx))/length(PDmat);
            ProbNeg1sig(indx) = sum(PDmat(indx)<CritNeg1sig(indx))/length(PDmat);
            ProbNeg2sig(indx) = sum(PDmat(indx)<CritNeg2sig(indx))/length(PDmat);
        end
        %CritMat = 
        %
        %%
        figure;
        plot(DelVec, ProbNeg2sig, 'r', DelVec, ProbNeg1sig, 'b', DelVec,ProbMean, 'k', DelVec, ProbPos1sig, ':b', DelVec, ProbPos2sig, ':r', 'LineWidth', 2);
       % legend( '$\Delta$ $\rho$ = 1 %', '$\Delta \rho = 2.5\%$', '$\Delta \rho = 5\%$', 'Location', 'Best', 'Interpreter','LaTex'); 
        xlabel('Disagreement ($\Delta$)', 'FontSize', 12, 'Interpreter','LaTex'); 
        ylabel('kurtosis of log PD ratio', 'FontSize', 12 );
        title([ 'NewBase Case ( $\rho$ = ',num2str(rho),' )'], 'FontSize', 16, 'Interpreter','LaTex', 'FontWeight', 'bold');
        hold off
        grid on
        %%
end





