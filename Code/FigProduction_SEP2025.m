%This file creates the figures in Section VII in the Online Appendix. The file uses pre-calculated results. 
%To create your own results you can ProductionSIMuncondV1_SEP2025.m to create the data.

clear; close all; clc;
%
DatUncon = load('Data/Model Disagreement/ProdUnconditionalDel04V2.mat'); %Change to del 0 and 08 here if you want to change to another delta. 
costVEC=DatUncon.costVEC;
corr1=DatUncon.corr1;
corr5=DatUncon.corr5;
corr10=DatUncon.corr10;

savefigures = 0;


base_font_size = 18;
%%Correlation Puzzle 
figure;
p1=plot(costVEC,corr10(:,1),'k', 'MarkerIndices', 1:50:length(corr10(:,1)),'LineWidth',2);
hold on;
p2=plot(costVEC,corr5(:,1),'--bo', 'MarkerIndices',1:50:length(corr5(:,1)),'LineWidth',2);
hold on;
p3=plot(costVEC,corr1(:,1),'-.rx', 'MarkerIndices',1:50:length(corr1(:,1)),'LineWidth',2);
hold off
legend([p3,p2,p1], 'one-year', 'five-year', 'ten-year', 'Location', 'Southeast')
xlabel('Adjustment Cost Parameter ($\kappa$)', 'Interpreter','LaTex', 'FontSize',base_font_size)
ylabel('Return-Consumption Correlation', 'Interpreter','LaTex', 'FontSize',base_font_size);
title('Correlation Puzzle', 'FontSize', base_font_size, 'Interpreter','LaTex', 'FontWeight', 'bold');
grid on
%%
if savefigures==1
    saveas(p3,'Data/Model Disagreement/FigProdCorrpuzzle.eps','epsc')
end


figure;
p3=plot(costVEC,DatUncon.DataAP(:,1),'k','LineWidth',2);
xlabel('Adjustment Cost Parameter ($\kappa$)', 'Interpreter','LaTex', 'FontSize',base_font_size)
ylabel('Unconditional Mean in bp', 'Interpreter','LaTex', 'FontSize',base_font_size);
%title('Stock Market Risk Premium $E\left(\lambda_t\right)$', 'Interpreter','LaTex', 'FontSize',base_font_size);
title('Stock Market Risk Premium $E\left[\tilde{\lambda} \right]$', 'Interpreter','LaTex', 'FontSize',base_font_size);
set(gca,'YTick', [5; 10; 15; 20; 25]/10000);            
set(gca,'YTickLabel',{'5.0', '10.0', '15.0', '20.0', '25.0'})        
grid on
%%
if savefigures==1
    saveas(p3,'Data/Model Disagreement/FigProdRP.eps','epsc')
end



%%
figure;
p3=plot(costVEC,DatUncon.DataAP(:,2),'k','LineWidth',2);
xlabel('Adjustment Cost Parameter ($\kappa$)', 'Interpreter','LaTex', 'FontSize',base_font_size)
ylabel('Unconditional Mean in %', 'Interpreter','LaTex', 'FontSize',base_font_size);
title('Stock Market Volatility $E\left[\tilde{\sigma}_{R}\right]$', 'Interpreter','LaTex', 'FontSize',base_font_size);
set(gca,'YTick', [3; 4; 5; 6; 7; 8; 9]/100);            
set(gca,'YTickLabel',{'3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0'})        
grid on
%%
if savefigures==1
    saveas(p3,'Data/Model Disagreement/FigProdStdevS.eps','epsc')
end

%%
figure;
p3=plot(costVEC,DatUncon.DataAP(:,3),'k','LineWidth',2);
xlabel('Adjustment Cost Parameter ($\kappa$)', 'Interpreter','LaTex', 'FontSize',base_font_size)
ylabel('Unconditional Mean', 'Interpreter','LaTex', 'FontSize',base_font_size);
title('Stock Market Demand Exposure $E\left[\tilde{\sigma}^{\alpha}_{R}\right]$', 'Interpreter','LaTex', 'FontSize',base_font_size);
grid on
%%
if savefigures==1
    saveas(p3,'Data/Model Disagreement/FigProdsigalpS.eps','epsc')
end


figure;
p3=plot(costVEC,DatUncon.DataAP(:,4),'k','LineWidth',2);
xlabel('Adjustment Cost Parameter ($\kappa$)', 'Interpreter','LaTex', 'FontSize',base_font_size)
ylabel('Unconditional Mean', 'Interpreter','LaTex', 'FontSize',base_font_size);
title('Consumption Growth Volatility $E\left[\tilde{\sigma}_{C}\right]$', 'Interpreter','LaTex', 'FontSize',base_font_size);
grid on
%%
if savefigures==1
    saveas(p3,'Data/Model Disagreement/FigProdStdevC.eps','epsc')
end

figure;
p3=plot(costVEC,DatUncon.DataAP(:,5),'k','LineWidth',2);
xlabel('Adjustment Cost Parameter ($\kappa$)', 'Interpreter','LaTex', 'FontSize',base_font_size)
ylabel('Unconditional Mean', 'Interpreter','LaTex', 'FontSize',base_font_size);
title('Consumption Demand Exposure $E\left[\tilde{\sigma}^{\alpha}_{C}\right]$', 'Interpreter','LaTex', 'FontSize',base_font_size);
grid on
%%
if savefigures==1
    saveas(p3,'Data/Model Disagreement/FigProdsigalpC.eps','epsc')
end
