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

	l2norm = zeros(size(vec_lambda));
	l1norm = zeros(size(vec_lambda));
	fprintf( 1, '   gamma       norm(x,1)    norm(A*x-b)\n' );
	fprintf( 1, '---------------------------------------\n' );
	
	for k = 1:length(vec_lambda)
		
		fprintf( 1, '%8.4e', vec_lambda(k) );
		
		cvx_begin
			variable x(n);
			minimize( norm(A*x-b)+vec_lambda(k)*norm(x,1) );
		cvx_end
		l1norm(k) = norm(x,1);
		l2norm(k) = norm(A*x-b);
		fprintf( 1, '   %8.4e   %8.4e\n', l1norm(k), l2norm(k) );
	end
	plot( l1norm, l2norm );
	xlabel( 'norm(x,1)' );
	ylabel( 'norm(A*x-b)' );
	grid on
	
	
	EGM = x;
end