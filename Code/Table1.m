clear all;
dataVolume = load('Data/StockVolumeNYSE.mat');
Volume=dataVolume.Volume;
dV = log(Volume(2:end)./Volume(1:end-1));
dateV =dataVolume.Dates(2:end);
%just smooth out the crazy obvervation

dV(end-17) = 0.5*(dV(end-18)+dV(end-16));
 %consistent with other data

% plot(dV)
% mean(dV)
% datestr(dataVolume.Dates(1))
% datestr(dataVolume.Dates(end))
% 
load('Data/ShillerAnnualData.mat');
datestr(ShillerAnnualData.Dates(1))

Rm = ShillerAnnualData.RealReturnSP500InclDiv;
Rm = Rm(18:end); %make it consistent with volume

DIV = ShillerAnnualData.RealDividendsSP500;
CONS = ShillerAnnualData.RealConsumption;
Y1 = ShillerAnnualData.Y1Real-1;
gDIV = DIV(18:end)./DIV(17:end-1)-1;
gC = CONS(18:end)./CONS(17:end-1)-1;
gY1 = Y1(18:end)-Y1(17:end-1);
DatesS = ShillerAnnualData.Dates(18:end);


%-4 if not c
startL = 3;
endL = length(Rm)-7;
Rm = Rm(startL:endL);
gDIV = gDIV(startL:endL);
gC = gC(startL:endL);
dV = dV(startL-1:endL-1);
gY1=gY1(startL:endL);
dateV = dateV(startL-1:endL-1);
DatesS = DatesS(startL:endL);



DAT = [Rm dV gDIV gC gY1];
corr(DAT)

%log Changes
DATl = log(1+DAT);

%turnover
dTurnover = DATl(:,2)-DATl(:,1);
DATl = [DATl(:,1:2) dTurnover DATl(:,3:end)];
DIMDAT = size(DATl);
T = length(DATl);

DATl5 = zeros(T-4,DIMDAT(2));
for i=1:T-4
    DATl5(i,:)  = sum(DATl(i:i+4,:),1);
end

DATl10 = zeros(T-9,DIMDAT(2));
for i=1:T-9
    DATl10(i,:)  = sum(DATl(i:i+9,:),1);
end


CORR = corr(DATl);
CORR5 = corr(DATl5);
CORR10 = corr(DATl10);



DATAtable1 = [CORR(4,1) CORR(5,1) CORR(4,2) CORR(5,2) CORR(4,end) CORR(5,end);CORR5(4,1) CORR5(5,1) CORR5(4,2) CORR5(5,2) CORR5(4,end) CORR5(5,end);CORR10(4,1) CORR10(5,1)  CORR10(4,2) CORR10(5,2) CORR10(4,end) CORR10(5,end)];



LatexInputTAB.data = DATAtable1;
LatexInputTAB.tableRowLabels = {'1 year','5 year','10 year'};
LatexInputTAB.dataFormat = {'%.2f'}; % uses three digit precision floating point for all data values
    % % Define how NaN values in input.tableData should be printed in the LaTex table:
LatexInputTAB.dataNanString = '  ';
% % Column alignment in Latex table ('l'=left-justified, 'c'=centered,'r'=right-justified):
LatexInputTAB.tableColumnAlignment = 'c';
LatexInputTAB.tableBorders = 0;   
latexTable2 = latexTable(LatexInputTAB);
fid = fopen('Data/Table1.tex', 'w');
fprintf(fid, '%s\n', latexTable2{:});
fclose(fid);
