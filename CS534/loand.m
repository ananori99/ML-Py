


parfor i = 1:5700
Table2{i,1} = regexprep( string(loandefaultv22{i,1}),'\<...-','')
end

