function Results=normalitytest(x)

% This function calculates p-values for ten well-known normality tests and gives the results both on display and as a matrix.

% Included tests are:Kolmogorov-Smirnov test (Limiting form (KS-Lim), Stephens Method (KS-S), Marsaglia Method  (KS-M), Lilliefors test (KS-L)), Anderson-Darling (AD) test, Cramer-Von Mises (CvM) test, Shapiro-Wilk (SW) test, Shapiro-Francia (SF) test, Jarque-Bera (JB) test, D’Agostino and Pearson (DAP) test.
  
% Enter the data as a row vector in the workspace

% Most tests run for n<900.
 
% example1: Normally distributed data
% x1=[66 53 154 76 73 118 106 69 87 84 41 33 78 56 55 35 44 135 75 58];
 
% example1: Disturbed data
% x2=[66 253 154 276 73 118 106 69 87 84 41 33 78 56 55 35 44 135 75 58];
 
% Alpha value can be changed as required
alpha=0.05;

% Please cite this work as: 
% Öner, M., & Deveci Kocakoç, Ý. (2017). JMASM 49: A Compilation of Some Popular Goodness of Fit Tests for Normal Distribution: Their Algorithms and MATLAB Codes (MATLAB). Journal of Modern Applied Statistical Methods, 16(2), 30. 
% Copyright (c) (2016) Öner, M., Deveci Kocakoc, I.
 
% KOLMOGOROV-SMIRNOV TEST- LIMITING FORM
 
n=length(x);
i=1:n;
y=sort(x);
fx=normcdf(zscore(y));
dplus=max(abs(fx-i/n));
dminus=max(abs(fx-(i-1)/n));
Dn=max(dplus,dminus);
KSz=sqrt(n)*Dn;
s=-20:1:20;
a=(-1).^s.*exp(-2*(s.*KSz).^2); 
pvalue=1-sum(a);
Results(1,1)=KSz;
Results(1,2)=pvalue;
 
% KOLMOGOROV-SMIRNOV TEST - STEPHENS MODIFICATION
 
dKSz=Dn*(sqrt(n)-0.01+0.85/sqrt(n));
 
if dKSz<0.775
    pvalue=0.15;
elseif dKSz<0.819
    pvalue=((0.10-0.15)/(0.819-0.775))*(dKSz-0.775)+0.15;
elseif dKSz<0.895
    pvalue=((0.05-0.10)/(0.895-0.819))*(dKSz-0.819)+0.10;
elseif dKSz<0.995
    pvalue=((0.025-0.05)/(0.995-0.895))*(dKSz-0.895)+0.05;
elseif dKSz<1.035
    pvalue=((0.01-0.025)/(1.035-0.995))*(dKSz-0.995)+0.025;
else
    pvalue=0.01;
end
Results(2,1)=dKSz;
Results(2,2)=pvalue;
 
% KOLMOGOROV-SMIRNOV TEST - MARSAGLIA METHOD
 
k=ceil(n*Dn);
m=2*k-1;
h=k-n*Dn;
 
Hmatrix=zeros(m,m);
 
for i=1:m-1
   for j=2:m
      if i-j+1>=0
      Hmatrix(i,j)=1/factorial(i-j+1);
    else
      Hmatrix(i,j)=0;
    end
    end
end
 
for i=1:m-1
    Hmatrix(i,1)=(1-h^i)/factorial(i);
end
 
Hmatrix(m,:)=fliplr(Hmatrix(:,1)');
 
if h<=0.5
Hmatrix(m,1)=(1 - 2*h^m)/factorial(m);
else
Hmatrix(m,1)=(1 - 2*h^m + max(0,2*h-1)^m)/factorial(m);
end
    lmax = max(eig(Hmatrix));
    Hmatrix = (Hmatrix./lmax)^n;
    pvalue = (1 - exp(gammaln(n+1) + n*log(lmax) - n*log(n)) * Hmatrix(k,k));
Results(3,1)=KSz;
Results(3,2)=pvalue;
 
% KOLMOGOROV-SMIRNOV TEST - LILLIEFORS MODIFICATION
 
% % P = [n D20 D15]
% P=[5 0.289 0.303;
%    6 0.269 0.281;
%    7 0.252 0.264;
%    8 0.239 0.250;
%    9 0.227 0.238;
%    10 0.217 0.228;
%    11 0.208 0.218;
%    12 0.200 0.210;
%    13 0.193 0.202;
%    14 0.187 0.196;
%    15 0.181 0.190;
%    16 0.176 0.184;
%    17 0.171 0.179;
%    18 0.167 0.175;
%    19 0.163 0.170;
%    20 0.159 0.166;
%    25 0.143 0.150;
%    30 0.131 0.138;
%    40 0.115 0.120;
%    100 0.074 0.077;
%    400 0.037 0.039;
%    900 0.025 0.026];
%  
% aaa=P(:,1)';
% subind=max(find(aaa<n));
% upind=subind+1;
% xxx=P(subind:upind,:);
%  
% if aaa(upind)==n
%    D20=xxx(2,2);
%    D15=xxx(2,3);
% else
%     D20=xxx(1,2)+(n-aaa(subind))*((xxx(2,2)-xxx(1,2))/(xxx(2,1)-xxx(1,1)));
%     D15=xxx(1,3)+(n-aaa(subind))*((xxx(2,3)-xxx(1,3))/(xxx(2,1)-xxx(1,1)));
% end
%  
% a1=-7.01256*(n+2.78019);
% b1=2.99587*sqrt(n+2.78019);
% c1=2.1804661+0.974598/sqrt(n)+1.67997/n;
%  
% a2=-7.90289126054*(n^0.98);
% b2=3.180370175721*(n^0.49);
% c2=2.2947256;
%  
% if n>100
%    D10=(-b2-sqrt(b2^2-4*a2*c2))/(2*a2);
%    a=a2;
%    b=b2;
%    c=c2;
% else
%    D10=(-b1-sqrt(b1^2-4*a1*c1))/(2*a1);
%    a=a1;
%    b=b1;
%    c=c1;
% end
%  
% if Dn==D10
%         pvalue=0.10;
%     elseif Dn>D10
%         pvalue=exp(a*Dn^2+b*Dn+c-2.3025851);
%     elseif Dn>=D15
%         pvalue=((0.10-0.15)/(D10-D15))*(Dn-D15)+0.15;
%     elseif Dn>=D20
%        pvalue=((0.15-0.20)/(D15-D20))*(Dn-D20)+0.20;
%     else
%        pvalue=0.20;
% end
Results(4,1)=0*Results(3,1);
Results(4,2)=0*Results(3,2);
 
% ANDERSON-DARLING TEST
 
adj=1+0.75/n+2.25/(n^2);
i=1:n;
ui=normcdf(zscore(y),0,1);
oneminusui=sort(1-ui);
lastt=(2*i-1).*(log(ui)+log(oneminusui));
asquare=-n-(1/n)*sum(lastt);
AD=asquare*adj;
 
if AD<=0.2
    pvalue=1-exp(-13.436+101.14*AD-223.73*AD^2);
elseif AD<=0.34
    pvalue=1-exp(-8.318+42.796*AD-59.938*AD^2);
elseif AD<=0.6
    pvalue=exp(0.9177-4.279*AD-1.38*AD^2);
elseif AD<=153.467
    pvalue=exp(1.2937*AD-5.709*AD+0.0186*AD^2);
else
    pvalue=0;
end
Results(5,1)=AD;
Results(5,2)=pvalue;
 
% CRAMER - VON MISES TEST
 
adj=1+0.5/n;
i=1:n;
fx=normcdf(zscore(y),0,1);
gx=(fx-((2*i-1)/(2*n))).^2;
CvMteststat=(1/(12*n))+sum(gx);
AdjCvM=CvMteststat*adj;
 
if AdjCvM<0.0275
    pvalue=1-exp(-13.953+775.5*AdjCvM-12542.61*(AdjCvM^2));
elseif AdjCvM<0.051
    pvalue=1-exp(-5.903+179.546*AdjCvM-1515.29*(AdjCvM^2));
elseif AdjCvM<0.092
    pvalue=exp(0.886-31.62*AdjCvM+10.897*(AdjCvM^2));
elseif AdjCvM>=0.093
    pvalue=exp(1.111-34.242*AdjCvM+12.832*(AdjCvM^2));
end
Results(6,1)=AdjCvM;
Results(6,2)=pvalue;
    
% SHAPIRO-WILK TEST
 
a=[];
i=1:n;
mi=norminv((i-0.375)/(n+0.25));
u=1/sqrt(n);
m=mi.^2;
 
a(n)=-2.706056*(u^5)+4.434685*(u^4)-2.07119*(u^3)-0.147981*(u^2)+0.221157*u+mi(n)/sqrt(sum(m));
a(n-1)=-3.58263*(u^5)+5.682633*(u^4)-1.752461*(u^3)-0.293762*(u^2)+0.042981*u+mi(n-1)/sqrt(sum(m));
a(1)=-a(n);
a(2)=-a(n-1);
eps=(sum(m)-2*(mi(n)^2)-2*(mi(n-1)^2))/(1-2*(a(n)^2)-2*(a(n-1)^2));
a(3:n-2)=mi(3:n-2)./sqrt(eps);
    ax=a.*y;
    KT=sum((x-mean(x)).^2);
    b=sum(ax)^2;
    SWtest=b/KT;
mu=0.0038915*(log(n)^3)-0.083751*(log(n)^2)-0.31082*log(n)-1.5861;
sigma=exp(0.0030302*(log(n)^2)-0.082676*log(n)-0.4803);
z=(log(1-SWtest)-mu)/sigma;
pvalue=1-normcdf(z,0,1);
Results(7,1)=SWtest;
Results(7,2)=pvalue;
 
% SHAPIRO-FRANCIA TEST
 
mi=norminv((i-0.375)/(n+0.25));
micarp=sqrt(mi*mi');
weig=mi./micarp;
pay=sum(y.*weig)^2;
payda=sum((y-mean(y)).^2);
SFteststa=pay/payda;
 
u1=log(log(n))-log(n);
u2=log(log(n))+2/log(n);
mu=-1.2725+1.0521*u1;
sigma=1.0308-0.26758*u2;
 
zet=(log(1-SFteststa)-mu)/sigma;
pvalue=1-normcdf(zet,0,1);
Results(8,1)=SFteststa;
Results(8,2)=pvalue;
 
% JARQUE-BERA TEST
 
E=skewness(y);
B=kurtosis(y);
JBtest=n*((E^2)/6+((B-3)^2)/24);
pvalue=1-chi2cdf(JBtest,2);
Results(9,1)=JBtest;
Results(9,2)=pvalue;
 
% D'AGOSTINO - PEARSON TEST
 
beta2=(3*(n^2+27*n-70)*(n+1)*(n+3))/((n-2)*(n+5)*(n+7)*(n+9));
wsquare=-1+sqrt(2*(beta2-1));
delta=1/sqrt(log(sqrt(wsquare)));
alfa=sqrt(2/(wsquare-1));
 
expectedb2=(3*(n-1))/(n+1);
varb2=(24*n*(n-2)*(n-3))/(((n+1)^2)*(n+3)*(n+5));
sqrtbeta=((6*(n^2-5*n+2))/((n+7)*(n+9)))*sqrt((6*(n+3)*(n+5))/(n*(n-2)*(n-3)));
A=6+(8/sqrtbeta)*(2/sqrtbeta+sqrt(1+4/(sqrtbeta^2)));
 
squarerootb=skewness(y);
Y=squarerootb*sqrt(((n+1)*(n+3))/(6*(n-2)));
zsqrtbtest=delta*log(Y/alfa+sqrt((Y/alfa)^2+1));
 
b2=kurtosis(y);
zet=(b2-expectedb2)/sqrt(varb2);
ztestb2=((1-2/(9*A))-((1-2/A)/(1+zet*sqrt(2/(A-4))))^(1/3))/sqrt(2/(9*A));
 
DAPtest=zsqrtbtest^2+ztestb2^2;
 
pvalue=1-chi2cdf(DAPtest,2);
Results(10,1)=DAPtest;
Results(10,2)=pvalue;
 
% Compare p-value to alpha
for i=1:10
    if Results(i,2)>alpha
        Results(i,3)=1;
    else
        Results(i,3)=0;
    end
end
 
% Output display
 
disp(' ') 
disp('Test Name                  Test Statistic   p-value   Normality (1:Normal,0:Not Normal)')
disp('-----------------------    --------------  ---------  --------------------------------')
fprintf('KS Limiting Form               %6.4f \t     %6.4f                 %1.0f \r',KSz,Results(1,2),Results(1,3))
fprintf('KS Stephens Modification       %6.4f \t     %6.4f                 %1.0f \r',dKSz,Results(2,2),Results(2,3))
fprintf('KS Marsaglia Method            %6.4f \t     %6.4f                 %1.0f \r',KSz,Results(3,2),Results(3,3))
fprintf('KS Lilliefors Modification     %6.4f \t     %6.4f                 %1.0f \r',Dn,Results(4,2),Results(4,3))
fprintf('Anderson-Darling Test          %6.4f \t     %6.4f                 %1.0f \r',AD,Results(5,2),Results(5,3))
fprintf('Cramer- Von Mises Test         %6.4f \t     %6.4f                 %1.0f \r',AdjCvM,Results(6,2),Results(6,3))    
fprintf('Shapiro-Wilk Test              %6.4f \t     %6.4f                 %1.0f \r',SWtest,Results(7,2),Results(7,3))
fprintf('Shapiro-Francia Test           %6.4f \t     %6.4f                 %1.0f \r',SFteststa,Results(8,2),Results(8,3))
fprintf('Jarque-Bera Test               %6.4f \t     %6.4f                 %1.0f \r',JBtest,Results(9,2),Results(9,3))
fprintf('DAgostino & Pearson Test       %6.4f \t     %6.4f                 %1.0f \r',DAPtest,Results(10,2),Results(10,3))
 



