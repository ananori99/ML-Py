%%Implement ElasticNet Regresion using Stochastic Gradient Descent

function b = elasticNet(lambda, alpha, learn, mini, Xr, yr)



%%SGD from slide 2, p27
%%also with refrence to the slides at:
%%http://www.stat.ucdavis.edu/~chohsieh/teaching/ECS289G_Fall2015/lecture3.pdf
t = 0;
%%you can change the tolerance used to check convergance here 
tolerance = 0.00000001;


%%"warm start" with a beta of ud val 0 to 1 
b = rand(size(Xr,2));
b0 = ones(size(Xr,2));



%%Then do SGD on the ridge portion
while norm(b - b0)/norm(b) > tolerance 
    %%this should not be true for the ones and uid [0,1]

%%use mini- batch size to randomly select which indices of Xraw and Yraw will be computed on at each itteration
indx=randsample(1:size(Xr,1),mini) ;
X = Xr(indx,:) ; %random rows of Xraw
y = yr(indx,:);%random rows of yraw


%%Compute the gradient for the ridge portion of elastic net
g = X.'*(X*b - y) + lambda.* alpha.* b;

%% Update beta according to the learning rate
b0=b;
b = b - learn.^t .* g;

%%the above would be the differentailble g(x) of proximal gradient descent, 
%%now map accoriding to slide 3, page 44:
 
b = prox(b, alpha);
t= t+1;
end 

end %end of elastic net
%% "use the continuous part (loss + l2) to calculate the gradient 
%%and then use the proximal mapping onto the l1 norm."

%%https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture18.pdf

function pb = prox(x,a)
%%the projector for proximal gradient descent for the l1 norm
for idx = 1:numel(x)
    xi = x(idx);
if (xi <= -a)
x(idx) = xi -a 
end
if (xi <= a)
x(idx) = 0;
end 
if (a <xi)
x(idx) = xi + a;
end
end %%end for
end %%end prox


