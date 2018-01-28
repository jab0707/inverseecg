%% HELP inverse_TotalVariation
%
%	This inverse method implements the Total Variation method.
%	It solves a leas-squares problem qith an l1 regularization term on the
%	first derivative.
%
%	The function assums that the user provides a matrix that estimates the
%	first derivatives.
%
%			INPUT (of inverse function):
% 					- A - <N M>double - forward operator of the linear system.
% 					- R - <H,M>double - regularization matrix / derivative estimation.
% 					- ECG - <N,T>double - Body surface measurements.
% 					- vec_lambda - double<1,NS> - vector of lambdas to tests.
%
%			OUTPUT:
% 					- EGM - <M,T>double - estimated TMP.
%
%			DEPENDENCES:
%				- CVX - http://cvxr.com/
%
%			AUTHOR:
%					Jaume Coll-Font <jcollfont@gmail.com>
%
%

function [EGM] = inverse_TotalVariation( A, R, ECG, vec_lambda)

	%% define
	[T] = size(ECG,2);
	[N,M] = size(A);
	[P] = size(R,1);
	
	vECG = reshape(ECG,[T*N,1]);
	
	%% for all lambdas
	x_init = zeros(M,T);
	l2norm = zeros(size(vec_lambda));
	l1norm = zeros(size(vec_lambda));
	EGM_lam = cell(1,numel(vec_lambda));
	
	for k = length(vec_lambda):-1:1
		
% 		kA = kron(speye(T),A);
% 		kR = kron(speye(T),R);
% 		cvx_begin 
% 			variable x_opt(M*T);
% 			minimize( norm( kA*x_opt - vECG ,2) + vec_lambda(k)*norm( kR*x_opt ,1) );
% 		cvx_end
		
		cvx_begin 
			variable x_opt(M,T);
			minimize( norm( A*x_opt - ECG ,'fro') + vec_lambda(k)*sum(sum( abs(R*x_opt) )) );
		cvx_end
		
% 		obj_fun  = @(x) TotalVariationObjective( x, A, ECG, sumI, R, M, P,T, vec_lambda(k));
% 		
% 		x_opt = fminunc(obj_fun, x_init, options);
		
		EGM_lam{k} = reshape(x_opt,[M,T]);
% 		x_init = x_opt;
		
		l1norm(k) = sum(sum(  abs(R*EGM_lam{k}) ));
		l2norm(k) = norm( A*EGM_lam{k} - ECG ,'fro' );
		
		
% 		if (length(vec_lambda) - k) > 3
% 			[~, kappa] = maxCurvatureLcurve(log([l1norm(k:end);l2norm(k:end)]), log10(vec_lambda(k:end)), min(numel(vec_lambda(k:end)), 10) );
% 		end
		
	end
	
	[lambdaCornerIX, kappa] = maxCurvatureLcurve(log([l1norm;l2norm]), log10(vec_lambda)/2,  max( round(numel(vec_lambda)/10) ,3) );
	
	EGM = EGM_lam{lambdaCornerIX};
	
end

% 
% function [cost] = TotalVariationObjective( x, A, ECG, sumI, R, M,P,T, lambda)
% 
% 	cost = norm( A*x - ECG ,'fro') + lambda * norm( reshape(R*x,[P*T,1]) ,1);
% 	
% % 	grad =  A'*A*x - 2*A'*ECG   +   lambda * sumI * ( sign(R*x).*repmat(R*ones(M,1),[1,T]) );
% 	
% end
