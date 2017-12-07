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


function [EGM_sol, lambdaCornerIX, EGM, rho, eta] = tikhonov_jcf(A, R, C, ECG, vec_lambda, underdetermined, frobenius)


	%% define
	numLam = numel(vec_lambda);
	[N,M] = size(A);
	T = size(ECG,2);
	
	doplots = false;	% unless desired do not do plots
	
	if numel(C) == 0
		C = eye(N);
	end
	
	if numel(R) == 0
		R = eye(M);
	end
	
	if ~exist('frobenius')
		frobenius = true;
	end
	
	%% set system of equations
	if underdetermined
		[M1,M2,M3,M4,y] = setUnderdeterminedEuations(A,R,C,ECG);
	else
		[M1,M2,M3,M4,y] = setOverdeterminedEuations(A,R,C,ECG);
	end
	
	
	%% Compute inverse solution for all lambda values 
	EGM = cell(1,numLam);
	rho = zeros(numLam,T);
	eta = zeros(numLam,T);
	for lam = 1:numLam
		
		EGM{lam} = solveInverseProblem(M1,M2,M3,y,vec_lambda(lam));
		
% 		rho(lam) = norm( ECG - A*EGM{lam}	, 'fro');
% 		eta(lam) = norm( R*EGM{lam}			, 'fro');
		
		rho(lam,:) = sum( ( ECG - A*EGM{lam}	).^2 , 1);
		eta(lam,:) = sum( ( R*EGM{lam}			).^2 , 1);
		
	end
	
	%% Choose to select independent lambda or single lambda
	EGM_sol = zeros(M,T);
	if frobenius
		
		% compute L-curve and select corner
		rho = sum(rho,2)';
		eta = sum(eta,2)';
		[lambdaCornerIX, kappa,splnApp] = maxCurvatureLcurve(log([rho;eta]), log10(vec_lambda), numel(vec_lambda));
		
		%% return solution
		EGM_sol = EGM{lambdaCornerIX};
		
	else
		% for every time instance
		for tt = 1:T
			
			% compute L-curve and select corner
			[lambdaCornerIX, kappa,splnApp] = maxCurvatureLcurve(log([rho(:,tt)';eta(:,tt)']), log10(vec_lambda), numel(vec_lambda));

			%% return solution
			EGM_sol(:,tt) = EGM{lambdaCornerIX}(:,tt);
			
		end
		
	end
	
	if doplots
		spacing = (vec_lambda(end) - vec_lambda(1))/numel(vec_lambda)*5;
		plotSamp = splnApp(linspace(vec_lambda(1)+spacing,vec_lambda(end) +spacing,1000));
		plot(plotSamp(1,:),plotSamp(2,:),'g-','LineWidth',2);
		hold on;
		scatter(log(rho), log(eta),30,kappa,'fill');
		plot(log(rho(lambdaCornerIX)), log(eta(lambdaCornerIX)),'ro');
		hold off;
	end
	
	

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
