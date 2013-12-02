function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
[J_old, grad_old] = costFunction(theta, X, y);
J = J_old + sum(theta(2:end).^2) * lambda / 2 / m;
grad = grad_old;
grad(2:end) = grad(2:end) + theta(2:end).*lambda./m;
% for i=1:m
%     J=J-y(i)*log(sigmoid(X(i,:)*theta))-(1-y(i))*log(1-sigmoid(X(i,:)*theta));
% end
% J=J/m+lambda/2/m*sum(theta.^2);
% 
% grad_temp=zeros(size(theta));
% grad_temp(1)=sum((sigmoid(X*theta)-y).*X(:,1))/m;
% for count=2:size(theta)
%     t=(sigmoid(X*theta)-y).*X(:,count);
%     grad_temp(count)=sum(t)/m+lambda/m*theta(count);
% end
% grad=grad_temp;
% 
end
