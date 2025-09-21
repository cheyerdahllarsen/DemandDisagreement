%This file creates the figures in Section VI in the Online Appendix. The file uses pre-calculated results. 
%To create your own results you can RunLearningCodeData_SEP2025.m to create the data.

clear;close all; clc;

dat1d4 = load('Data/Model Disagreement/Run_ForConferencesNOV18_d4_N1.mat');
dat2d4 = load('Data/Model Disagreement/Run_ForConferencesNOV18_d4_N2.mat');
dat3d4 = load('Data/Model Disagreement/Run_ForConferencesNOV18_d4_N3.mat');

dat1d2 = load('Data/Model Disagreement/Run_ForConferencesNOV18_d2_N1.mat');
dat2d2 = load('Data/Model Disagreement/Run_ForConferencesNOV18_d2_N2.mat');
dat3d2 = load('Data/Model Disagreement/Run_ForConferencesNOV18_d2_N3.mat');

dat1d0 = load('Data/Model Disagreement/Run_ForConferencesNOV18_d0_N1.mat');
dat2d0 = load('Data/Model Disagreement/Run_ForConferencesNOV18_d0_N2.mat');
dat3d0 = load('Data/Model Disagreement/Run_ForConferencesNOV18_d0_N3.mat');

VbarVEC = dat1d4.VbarVEC;


%consumption share dynamics
f0 = mean([dat1d0.fSTAT(1,:);dat2d0.fSTAT(1,:);dat3d0.fSTAT(1,:)]);
f2 = mean([dat1d2.fSTAT(1,:);dat2d2.fSTAT(1,:);dat3d2.fSTAT(1,:)]);
f4 = mean([dat1d4.fSTAT(1,:);dat2d4.fSTAT(1,:);dat3d4.fSTAT(1,:)]);

f_nolearning = [0.602 0.504 0.438]; %this is based on a longer simulation
f4(1) = f_nolearning(3);
%DELTA
DEL0 = mean([dat1d0.DEL_STAT(1,:);dat2d0.DEL_STAT(1,:);dat3d0.DEL_STAT(1,:)]);
DEL2 = mean([dat1d2.DEL_STAT(1,:);dat2d2.DEL_STAT(1,:);dat3d2.DEL_STAT(1,:)]);
DEL4 = mean([dat1d4.DEL_STAT(1,:);dat2d4.DEL_STAT(1,:);dat3d4.DEL_STAT(1,:)]);

VDEL0 =  mean([dat1d0.DEL_STAT(2,:);dat2d0.DEL_STAT(2,:);dat3d0.DEL_STAT(2,:)]);
VDEL2 =  mean([dat1d2.DEL_STAT(2,:);dat2d2.DEL_STAT(2,:);dat3d2.DEL_STAT(2,:)]);
VDEL4 =  mean([dat1d4.DEL_STAT(2,:);dat2d4.DEL_STAT(2,:);dat3d4.DEL_STAT(2,:)]);

r0 = mean([dat1d0.r_STAT(1,:);dat2d0.r_STAT(1,:);dat3d0.r_STAT(1,:)]);
r2 = mean([dat1d2.r_STAT(1,:);dat2d2.r_STAT(1,:);dat3d2.r_STAT(1,:)]);
r4 = mean([dat1d4.r_STAT(1,:);dat2d4.r_STAT(1,:);dat3d4.r_STAT(1,:)]);

rp0 = mean([dat1d0.rpt_STAT(1,:);dat2d0.rpt_STAT(1,:);dat3d0.rpt_STAT(1,:)]);
rp2 = mean([dat1d2.rpt_STAT(1,:);dat2d2.rpt_STAT(1,:);dat3d2.rpt_STAT(1,:)]);
rp4 = mean([dat1d4.rpt_STAT(1,:);dat2d4.rpt_STAT(1,:);dat3d4.rpt_STAT(1,:)]);

stdR0 = mean([dat1d0.stdRM_STAT(1,:);dat2d0.stdRM_STAT(1,:);dat3d0.stdRM_STAT(1,:)]);
stdR2 = mean([dat1d2.stdRM_STAT(1,:);dat2d2.stdRM_STAT(1,:);dat3d2.stdRM_STAT(1,:)]);
stdR4 = mean([dat1d4.stdRM_STAT(1,:);dat2d4.stdRM_STAT(1,:);dat3d4.stdRM_STAT(1,:)]);


base_font_size=18;

savefigures = 0;

skipdot = 2;
%FIG1
figure;
p1=plot(VbarVEC,f0,'k', 'MarkerIndices', 1:skipdot:length(f0),'LineWidth',2);
hold on;
p2=plot(VbarVEC,f2,'--bo', 'MarkerIndices',1:skipdot:length(f2),'LineWidth',2);
hold on;
p3=plot(VbarVEC,f4,'-.rx', 'MarkerIndices',1:skipdot:length(f4),'LineWidth',2);
hold off
axis([0 100 0.4 0.75]);
%legend('$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8', 'Interpreter','LaTex' ,'Location', 'Southeast');
legend([ p1, p2, p3 ],  {'$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8'},  'Location', 'Southeast', 'Interpreter','LaTex', 'FontSize', base_font_size)
xlabel('Prior Variance $V$', 'Interpreter','LaTex', 'FontSize',base_font_size)
ylabel('Unconditional Mean', 'Interpreter','LaTex', 'FontSize',base_font_size);
title('Consumption Share', 'FontSize', base_font_size, 'FontWeight', 'bold');
grid on
if savefigures==1
    saveas(p3,'Data/Model Disagreement/FigLearning_f_v1.eps','epsc')
end

%FIG2
figure;
p1=plot(VbarVEC,DEL0,'k', 'MarkerIndices', 1:skipdot:length(f0),'LineWidth',2);
hold on;
p2=plot(VbarVEC,DEL2,'--bo', 'MarkerIndices',1:skipdot:length(f2),'LineWidth',2);
hold on;
p3=plot(VbarVEC,DEL4,'-.rx', 'MarkerIndices',1:skipdot:length(f4),'LineWidth',2);
hold off
%legend('$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8', 'Interpreter','LaTex' ,'Location', 'Northeast');
legend([ p1, p2, p3 ],  {'$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8'},  'Location', 'Northeast', 'Interpreter','LaTex', 'FontSize', base_font_size)
xlabel('Prior variance $V$', 'Interpreter','LaTex', 'FontSize',base_font_size)
ylabel('Unconditional Mean', 'Interpreter','LaTex', 'FontSize',base_font_size);
title('Endogeneous Disagreement', 'FontSize', base_font_size, 'FontWeight', 'bold');
grid on
if savefigures==1
    saveas(p3,'Data/Model Disagreement/FigLearning_DeltaM_v1.eps','epsc')
end

%FIG3
figure;
p1=plot(VbarVEC,VDEL0,'k', 'MarkerIndices', 1:skipdot:length(f0),'LineWidth',2);
hold on;
p2=plot(VbarVEC,VDEL2,'--bo', 'MarkerIndices',1:skipdot:length(f2),'LineWidth',2);
hold on;
p3=plot(VbarVEC,VDEL4,'-.rx', 'MarkerIndices',1:skipdot:length(f4),'LineWidth',2);
hold off
%legend('$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8', 'Interpreter','LaTex' ,'Location', 'Southeast');
legend([ p1, p2, p3 ],  {'$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8'},  'Location', 'Southeast', 'Interpreter','LaTex', 'FontSize', base_font_size)
xlabel('Prior Variance $V$', 'Interpreter','LaTex', 'FontSize',base_font_size)
ylabel('Uncondtional Standard Deviation', 'Interpreter','LaTex', 'FontSize',base_font_size);
title('Endogeneous Disagreement', 'FontSize', base_font_size, 'FontWeight', 'bold');
grid on
if savefigures==1
    saveas(p3,'Data/Model Disagreement/FigLearning_DeltaSTD_v1.eps','epsc')
end


%FIG4
figure;
p1=plot(VbarVEC,r0-0.033^2+0.02^2,'k', 'MarkerIndices', 1:skipdot:length(f0),'LineWidth',2);
hold on;
p2=plot(VbarVEC,r2-0.033^2+0.02^2,'--bo', 'MarkerIndices',1:skipdot:length(f2),'LineWidth',2);
hold on;
p3=plot(VbarVEC,r4-0.033^2+0.02^2,'-.rx', 'MarkerIndices',1:skipdot:length(f4),'LineWidth',2);
hold off
%legend('$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8', 'Interpreter','LaTex' ,'Location', 'Northeast');
legend([ p1, p2, p3 ],  {'$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8'},  'Location', 'Northeast', 'Interpreter','LaTex', 'FontSize', base_font_size)
xlabel('Prior Variance $V$', 'Interpreter','LaTex', 'FontSize',base_font_size)
ylabel('Unconditional Mean ', 'Interpreter','LaTex', 'FontSize',base_font_size);
title('Risk-free Interest Rate', 'FontSize', base_font_size, 'FontWeight', 'bold');
grid on
if savefigures==1
    saveas(p3,'Data/Model Disagreement/Fig_learning_r_v1.eps','epsc')
end



%FIG5
figure;
p1=plot(VbarVEC,rp0+0.033^2,'k', 'MarkerIndices', 1:skipdot:length(f0),'LineWidth',2);
hold on;
p2=plot(VbarVEC,rp2+0.033^2,'--bo', 'MarkerIndices',1:skipdot:length(f2),'LineWidth',2);
hold on;
p3=plot(VbarVEC,rp4+0.033^2,'-.rx', 'MarkerIndices',1:skipdot:length(f4),'LineWidth',2);
hold off
%legend('$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8' , 'Interpreter','LaTex','Location', 'Northeast');
legend([ p1, p2, p3 ],  {'$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8'},  'Location', 'Northeast', 'Interpreter','LaTex', 'FontSize', base_font_size)
xlabel('Prior Variance $V$', 'Interpreter','LaTex', 'FontSize',base_font_size)
ylabel('Unconditional Mean in bp', 'Interpreter','LaTex', 'FontSize',base_font_size);
title('Stock Market Risk Premium', 'FontSize', base_font_size, 'FontWeight', 'bold');
set(gca,'YTick', [10; 20; 30; 40; 50; 60; 70; 80; 90; 100]/10000);            
set(gca,'YTickLabel',{'10'; '20'; '30'; '40'; '50'; '60'; '70'; '80'; '90'; '100'})        
grid on
if savefigures==1
    saveas(p3,'Data/Model Disagreement/Fig_learning_RiskP_v1.eps','epsc')
end


%FIG6
figure;
p1=plot(VbarVEC,stdR0+(0.033-0.02),'k', 'MarkerIndices', 1:skipdot:length(f0),'LineWidth',2);
hold on;
p2=plot(VbarVEC,stdR2+(0.033-0.02),'--bo', 'MarkerIndices',1:skipdot:length(f2),'LineWidth',2);
hold on;
p3=plot(VbarVEC,stdR4+(0.033-0.02),'-.rx', 'MarkerIndices',1:skipdot:length(f4),'LineWidth',2);
hold off
%legend('$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8', 'Interpreter','LaTex' ,'Location', 'Northeast');
legend([ p1, p2, p3 ],  {'$\Delta$ = 0', '$\Delta$ = 0.4', '$\Delta$ = 0.8'},  'Location', 'Northeast', 'Interpreter','LaTex', 'FontSize', base_font_size)
xlabel('Prior Variance $V$', 'Interpreter','LaTex', 'FontSize',base_font_size)
ylabel('Unconditional Mean', 'Interpreter','LaTex', 'FontSize',base_font_size);
title('Stock Market Volatility ', 'FontSize', base_font_size, 'FontWeight', 'bold');
grid on
if savefigures==1
    saveas(p3,'Data/Model Disagreement/Fig_learning_StdevRM_v1.eps','epsc')
end


