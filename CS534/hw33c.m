%generate predictions from the 100 bootstraps by averaging
pred=[];
X= table2array(loandefaultTestCopy);
%size(X);
for i=1:100
    %the log function
    yhat =(1/(1+ exp(-X*allCoef(:,i))))';
   
    pred = [pred, yhat];


end
%average 
M = mean(pred,2);
%the AUC
  [~,~,~,AUC] = perfcurve(table2array(loandefault2),M,0)