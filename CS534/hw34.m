function sgdTree(M,v,q, X,Y)
f= ones(size(X,1);
%subsample at rate q
rate = int(q*size(X,1));

    s = randperm(size(X,1));
    boot = data(s(1:rate),:);
    yboot =  out(s(1:rate),:);
    
    
for m = 1:M
    %Compute gradient residual
    r = -gradient(yboot-boot*f)

end

