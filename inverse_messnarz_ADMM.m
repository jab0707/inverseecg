
%% HELP:
% 
% 				This function implements an algorithm that solves the inverse problem
% 				published by Messnarz et al. in TBME-IEEE:
% 					"A New Spatiotemporal Regularization Approach
% 					for Reconstruction of Cardiac Transmembrane
% 					Potential Patterns"
% 
% 				Messnarz et al. pos e least squares minimization problem
% 				defined as:
% 
% 					min_X | LX - Y |_2^2 + lambda*|RX|_2^2
% 						st. x(:,t) - x(:,t+1) <= 0
% 							x(:,1) >= minX
% 							x(:,end) <= maxX
% 
% 				where L is the forward matrix, R is the regularization matrix,
% 				X are the transmembrane potentials (TMP) and y are the
% 				ECG recordings.
% 
% 				This problem is equivalent to a quadratic problem of the form:
% 					min_x  x'Gx  +  x'c + l
% 						st. Ax >= b
% 
% 					where	G = (L'L + lambdaR'R)
% 							c = -L'Y
% 
% 				The practical implementation of this algorithm is done through
% 				the ADMM minimization of the function:
% 
% 					min_{x,z} f(x) + g(z)  + 
% 					min_{x,z} \sum_{i=1}^T \|Ax_i - y_i\|_2^2 + \sum_{j=1}^T proj( z_j ) + proj(1- z_{T+1}) 
% 
% 						st. x_{t} - x_{t-1} = z_t  ; t=[2..T]
% 							x_1 = z_1
% 							x_{T} = z_{T+1}
% 
% 						where proj is defined as: { g(z<0)=infty, g(z>=0)=0 }
% 
% 				Thus the objective the augmented lagrangian is:
% 
% 					min_{x,z,w} L(x,z,w) =
% 								\sum_{i=1}^T \|Ax_i - y_i\|_2^2 + proj( z ) + w^T( Gx(:) -b -z) + \rho/2\|Gx(:) -b - z\|_2^2
% 
% 				ADMM will solve this augmented lagrangian minimization by
% 				optimizing recursively for x, z and w (lamk in the code).
% 
% 				rho: rho determines how important is the augmented term in each
% 				iteration. It has effects on the convergence rate but and is
% 				problem dependent. Its value is updated every revisit_rho
% 				iterations if the residuals are 
%
%			INPUT (of inverse function):
% 					- A - <N M>double - forward operator of the linear system.
% 					- R - <H,M>double - regularization matrix.
% 					- ECG - <N,T>double - Body surface measurements.
% 					- lambda - double - regularization term.
% 					- initialx - <M,T>double - initial guess for the TMP
% 					solution.
% 					- rho - double - augmented term weight.
% 					- min_r - double - stopping criteria for the primal
% 					residual.
% 					- min_s - double - stopping criteria for the dual residual.
%					- margin - <2,1>double - minimum and maximum bounds for
%					the transmembrane potentials.
% 					- verbose - boolean - print ADMM iteratinons information.
%
%			OUTPUT:
% 					- xk - <M,T>double - estimated TMP.
% 					- zk - <M,T+1>double - estimated slack variable.
%
%			DEPENDENCES:
%
%			AUTHOR:
%					Jaume Coll-Font <jcollfont@gmail.com>
%


function [EGM_sol, lambda_corner, xk] = inverse_messnarz_ADMM(input_data, A, R, vector_lambda, margins, ADMM_params)


	%% DEFINES
		verbose = true;
	
	% default upper and lower bounds for TMPS
		maxBound = 35;	% mV
		minBound = -85;	% mV
		
	% default ADMM params
		rho_ADMM = 10;
		min_r = 1e-5;
		min_s = 1e-5;

	% vector of regularization parameters
		num_lambda = numel(vector_lambda);
			
	% adapt parameters
		if exist('ADMM_params')
			rho_ADMM = ADMM_params(3);
			min_r = ADMM_params(1);
			min_s = ADMM_params(2);
			maxBound = margins(2);
			minBound = margins(1);
		end
		if exist('margins')
			maxBound = margins(2);
			minBound = margins(1);
		end		
		
	% initialize data 
		[N, T] = size(input_data);
		M = size(A,2);
		xk = cell(1,num_lambda);
		logRes = zeros(1,num_lambda);
		logReg = zeros(1,num_lambda);
	
	% initial guess
		initialx = repmat(linspace(minBound,maxBound,T),M,1);
		zk = [];
		lamk = [];

	%% for each lambda
		tic
		for lam = num_lambda:-1:1

			if verbose
				fprintf('Solving inverse problem for lambda %0.4f\n',log10(vector_lambda(lam)) );
			end
			
			% ADMM solver
			[xk{lam}, zk, rho_ADMM, lamk] = solve_messnarz_ADMM(	A,...			forward matrix
														R,...					regularization matrix
														input_data,...			input data
														vector_lambda(lam),...	current lambda
														initialx,...			initial guess
														rho_ADMM,...					rho (ADMM)
														min_r,...				stopping criteria residual
														min_s,...				stopping criteria gradient
														[minBound maxBound],...	potential bounds
														false,...				verbose
														zk,...				`	feedback of zk for warm start
														lamk...					feedback of lamk for warm start
														);
			
			rho(lam) = (norm(input_data - A*xk{lam},'fro'));
			eta(lam) = (norm(R*xk{lam},'fro'));

			% initialize next
			initialx = xk{lam};

		end
		toc
		
	%% Select corner
		[lambda_corner] = maxCurvatureLcurve(log([rho;eta]), log10(vector_lambda), 8);
		
	%% return solutions
		EGM_sol = xk{lambda_corner};

end
