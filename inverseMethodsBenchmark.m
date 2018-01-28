%% HELP:
%
%		This function computes multiple inverse solutions as a benchmark
%		analysis.
%		The inverse methods tested are:
%
%			- Tikhonov 0th order.
%			- Tikhonov 1st order
%			- Tikhonov 0th order
%			- Total Variation
%			- Greensite w/ Tikhonov
%			- Splines inverse w/ Tikhonov 0th order (or provided matrix)
%			- TSVD
%			ToDo:
%				- MFS
%				- GMRES
%
%
%		INPUT:
%			- A - <N,M>double - forward matrix.
%			- ECG - <N,T>double - measured ECG
%			- inputOptions - struct - each entry in the struct contains the
%			specific options of every inverse method:
%					- Tikh0 -> 
%							- lambdas - <int> - lambda range [min,max,num]
%							- C - <N,N>double - (OPTIONAL) covariance
%												matrix.
%							- frobenius - bool - (OPTIONAL) compute joint
%												L-curve for all time instances?
%					- Tikh1 -> 
%							- lambdas - <int> - lambda range [min,max,num]
%							- D - <P,M>double - gradient estimation (reg) matrix.
%							- C - <N,N>double - (OPTIONAL) covariance
%												matrix.
%							- frobenius - bool - (OPTIONAL) compute joint
%												L-curve for all time instances?
%					- Tikh2 ->
%							- lambdas - <int> - lambda range [min,max,num]
%							- Lapl - <P,M>double - laplacian estimation (reg) matrix.
%							- C - <N,N>double - (OPTIONAL) covariance
%												matrix.
%							- frobenius - bool - (OPTIONAL) compute joint
%												L-curve for all time instances?
%					- TSVD -> 
%							- C - <N,N>double - (OPTIONAL) covariance
%												matrix.
%							- frobenius - bool - (OPTIONAL) compute joint
%												L-curve for all time instances?
%					- TV -> 
%							- lambdas - <int> - lambda range [min,max,num]
%							- D - <P,M>double - gradient estimation (reg) matrix.
%							- frobenius - bool - (OPTIONAL) compute joint
%												L-curve for all time instances?
%					- Greensite -> 
%							- trunkSVD -<Q,1>int- truncation points to be
%													tested.
%							- lambdas - <int> - lambda range [min,max,num]
%							- R - <P,M>double - regularization matrix.
%							- C - <N,N>double - (OPTIONAL) covariance
%												matrix.
%							- frobenius - bool - (OPTIONAL) compute joint
%												L-curve for all time instances?
%					- Splines -> 
%							- lambdas - <int> - lambda range [min,max,num]
%							- D - <P,M>double - gradient estimation (reg) matrix.
%							- C - <N,N>double - (OPTIONAL) covariance
%												matrix.
%							- frobenius - bool - (OPTIONAL) compute joint
%												L-curve for all time instances?
%
%		OUTPUT:
%			- inverseOutput - struct - all the inverse solutions obtained. 
%					- Tikh0 -> 
%							- EGM - <N,M>double - inverse solution.
%							- lambda_opt - double - regularization
%													parameter.
%					- Tikh1 -> 
%							- EGM - <N,M>double - inverse solution.
%							- lambda_opt - double - regularization
%													parameter.
%					- Tikh2 ->
%							- EGM - <N,M>double - inverse solution.
%							- lambda_opt - double - regularization
%													parameter.
%					- TSVD -> 
%							- EGM - <N,M>double - inverse solution.
%							- lambda_opt - double - regularization
%													parameter.
%					- TV -> 
%							- EGM - <N,M>double - inverse solution.
%							- lambda_opt - double - regularization
%													parameter.
%					- Greensite -> 
%							- EGM - <N,M>double - inverse solution.
%							- lambda_opt - double - regularization
%													parameter.
%							- trunk_opt - int - truncation point of SVD.
%					- Splines -> 
%							- EGM - <N,M>double - inverse solution.
%							- lambda_opt - double - regularization
%													parameter.
%
%
%		AUTHOR:
%			Jaume Coll-Font <jcollfont@gmail.com>
%
%

function [inverseOutput] = inverseMethodsBenchmark( A, ECG, inputOptions )

	%% general params
	[~,T] = size(ECG);
	[N,M] = size(A);
	
	inverseOutput = inputOptions;
	
	%% 0th order Tikhonov
	if isfield( inputOptions,'Tikh0')
		
		% params
		vec_lambda = 10.^(2.*linspace(inputOptions.Tikh0.lambdas(1),inputOptions.Tikh0.lambdas(2),inputOptions.Tikh0.lambdas(3)));
		
		if ~isfield(inputOptions.Tikh0, 'frobenius')
			inverseOutput.Tikh0.frobenius = true;
		else
			inverseOutput.Tikh0.frobenius = inputOptions.Tikh0.frobenius;
		end
		if ~isfield(inputOptions.Tikh0, 'underdetermined')
			inverseOutput.Tikh0.underdetermined = false;
		else
			inverseOutput.Tikh0.underdetermined = inputOptions.Tikh0.underdetermined;
		end
		
		% compute
		[EGM_sol, lambdaCornerIX] = tikhonov_jcf( A, eye(M), eye(N), ECG, vec_lambda, inverseOutput.Tikh0.underdetermined, inverseOutput.Tikh0.frobenius);
		
		% output
		inverseOutput.EGM = EGM_sol;
		inverseOutput.lambda_opt = vec_lambda(lambdaCornerIX);
	end
	
	%% 1st order Tikhonov
	if isfield( inputOptions,'Tikh1')
		
		% params
		vec_lambda = 10.^(2.*linspace(inputOptions.Tikh1.lambdas(1),inputOptions.Tikh1.lambdas(2),inputOptions.Tikh1.lambdas(3)));
		
		if ~isfield(inputOptions.Tikh1, 'frobenius')
			inverseOutput.Tikh1.frobenius = true;
		else
			inverseOutput.Tikh1.frobenius = inputOptions.Tikh1.frobenius;
		end
		if ~isfield(inputOptions.Tikh1, 'underdetermined')
			inverseOutput.Tikh1.underdetermined = false;
		else
			inverseOutput.Tikh1.underdetermined = inputOptions.Tikh1.underdetermined;
		end
		
		% compute
		[EGM_sol, lambdaCornerIX] = tikhonov_jcf( A, inputOptions.Tikh1.D, eye(N), ECG, vec_lambda, inverseOutput.Tikh1.underdetermined, inverseOutput.Tikh1.frobenius);
		
		% output
		inverseOutput.EGM = EGM_sol;
		inverseOutput.lambda_opt = vec_lambda(lambdaCornerIX);
	end
	
	%% 2nd order Tikhonov
	if isfield( inputOptions,'Tikh2')
		
		% params
		vec_lambda = 10.^(2.*linspace(inputOptions.Tikh2.lambdas(1),inputOptions.Tikh2.lambdas(2),inputOptions.Tikh2.lambdas(3)));
		
		if ~isfield(inputOptions.Tikh2, 'frobenius')
			inverseOutput.Tikh2.frobenius = true;
		else
			inverseOutput.Tikh2.frobenius = inputOptions.Tikh2.frobenius;
		end
		if ~isfield(inputOptions.Tikh2, 'underdetermined')
			inverseOutput.Tikh2.underdetermined = false;
		else
			inverseOutput.Tikh2.underdetermined = inputOptions.Tikh2.underdetermined;
		end
		
		% compute
		[EGM_sol, lambdaCornerIX] = tikhonov_jcf( A, inputOptions.Tikh2.Lapl, eye(N), ECG, vec_lambda, inverseOutput.Tikh2.underdetermined, inverseOutput.Tikh2.frobenius);
		
		% output
		inverseOutput.EGM = EGM_sol;
		inverseOutput.lambda_opt = vec_lambda(lambdaCornerIX);
	end
	
	%% TSVD
	if isfield( inputOptions,'TSVD')
		
		% compute
		[EGM_sol, lambdaCornerIX] = TSVD_inverse( A, ECG, vec_lambda );
		
		% output
		inverseOutput.EGM = EGM_sol;
		inverseOutput.lambda_opt = vec_lambda(lambdaCornerIX);
	end



end












