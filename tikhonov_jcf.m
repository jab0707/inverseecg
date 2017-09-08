%% HELP:
%
%       [EGM_sol, lambdaCornerIX, EGM, rho, eta] = tikhonov_jcf(A, R, C, ECG, vec_lambda, underdetermined)
%
%		This function computes the inverse problem using Tikhonov
%		regularization with an arbitrary regularization matrix.
%
%		The function computes the overdetermined and underdetermined system
%		of equations and returns the inverse operator.
%
%


function [EGM_sol, lambdaCornerIX, EGM, rho, eta] = tikhonov_jcf(A, R, C, ECG, vec_lambda, underdetermined)


	%% define
	numLam = numel(vec_lambda);
	[N,M] = size(A);
	
	if numel(C) == 0
		C = eye(N);
	end
	
	if numel(R) == 0
		R = eye(M);
	end
	
	%% set system of equations
	if underdetermined
		[M1,M2,M3,M4,y] = setUnderdeterminedEuations(A,R,C,ECG);
	else
		[M1,M2,M3,M4,y] = setOverdeterminedEuations(A,R,C,ECG);
	end
	
	
	%% Compute L-curve
	EGM = cell(1,numLam);
	rho = zeros(1,numLam);
	eta = zeros(1,numLam);
	for lam = 1:numLam
		
		EGM{lam} = solveInverseProblem(M1,M2,M3,y,vec_lambda(lam));
		
		rho(lam) = norm( ECG - A*EGM{lam}	, 'fro');
		eta(lam) = norm( R*EGM{lam}			, 'fro');
		
	end
	
	%% Select corner
	[lambdaCornerIX, kappa] = maxCurvatureLcurve(log([rho;eta]), log10(vec_lambda), 8);
	plot(log(eta), log(rho),'x-');hold on;plot(log(eta(lambdaCornerIX)), log(rho(lambdaCornerIX)),'ro');hold off;
	pause(1);
	
	%% return solution
	EGM_sol = EGM{lambdaCornerIX};

end

%% set underdetermined matrices for the equation
%     // OPERATE ON DATA:
%     // Compute X = (R * R^T)^-1 * A^T (A * (R*R^T)^-1 * A^T + LAMBDA * (C*C^T)^-1 ) * Y
%     //         X = M3                *              G^-1                                 * (M4) * Y
%     // Will set:
%       //      M1 = A * (R*R^T)^-1 * A^T
%       //      M2 = (C*C^T)^-1
%       //      M3 = (R * R^T)^-1 * A^T
%       //      M4 = identity
%       //      y = measuredData
%     //.........................................................................
function [M1,M2,M3,M4,y] = setUnderdeterminedEuations(A,R,C,ECG)

	iRR = pinv(R*R');%inv(R*R');
	M1 = A*iRR*A';
	M2 = inv(C*C');
	M3 = iRR*A';
	M4 = eye(size(iRR));
	y = ECG;
	
end

%% set Overdetermined matrices for the equations
%         //.........................................................................
%         // OPERATE ON DATA:
%         // Computes X = (A^T * C^T * C * A +  LAMBDA * R^T * R) * A^T * C^T * C * Y
%         //          X = (M3)       *              G^-1                  * M4            * Y
%         //.........................................................................
%         // Will set:
%         //      M1 = A * C^T*C * A^T
%         //      M2 = R^T*R
%         //      M3 = identity
%         //      M4 = A^TC^TC
%         //      y = A^T * C^T*C * measuredData
%         //.........................................................................
function [M1,M2,M3,M4,y] = setOverdeterminedEuations(A,R,C,ECG)

	CC = C'*C;
	M1 = A'*CC*A;
	M2 = R'*R;
	M3 = eye(size(M2));
	M4 = A'*CC;
	y = M4*ECG;
	
end


%% Compute Inverse Solution
%     //............................
%     //
%     //      G = (M1 + lambda * M2)
%     //      b = G^-1 * y
%     //      x = M3 * b
%     //
%     //      A^-1 = M3 * G^-1 * M4
%     //...........................................................................................................
function [x] = solveInverseProblem(M1,M2,M3,y,lambda)

	G = (M1 + lambda*M2);
	b = G\y;
	x = M3*b;
	
end
