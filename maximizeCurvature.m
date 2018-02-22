function [lambda_IX] = maximizeCurvature(splnApprx, lambdaInit, rho, logLambdas)

	K = numel(splnApprx);
	
	lb = repmat(min(logLambdas),[K,1]);
	ub = repmat(max(logLambdas),[K,1]);
	
	[lambdaInit] = admm_optimization( lambdaInit, rho, splnApprx, lb, ub);
	
	objFcn = @(alpha) jointObjective( alpha, rho, splnApprx);
	
	opti_settings = optimset('Display','iter','MaxFunEvals',1e10,'Maxiter',1e10,'TolX',1e-10,'TolFun',1e-10,'LargeScale','off');
	
	[alpha, fval, extraflag] = fmincon( objFcn,  lambdaInit, [],[],[],[],lb,ub ,[],opti_settings);
	
	lambda_IX = zeros(1,K);
	for k = 1:K
		[~, lambda_IX(k)] = min(abs(alpha(k) - logLambdas));
	end
	
end


function [accCurv] = jointObjective( alpha, rho, splnApprx)

	K = numel(splnApprx);
	
	accCurv  = -splnApprx{1}(alpha(1));
	
	for k = 2:K
		
		accCurv = accCurv - splnApprx{k}(alpha(k)) + rho*( alpha(k) - alpha(k-1) )^2;
		
	end
end

function [cost] = separateObjective( alpha, t, alphas, rho, splnApprx)

	if t==1
		cost = -splnApprx(alpha);
	elseif t<numel(alphas)
		cost = -splnApprx(alpha) + rho*( alpha - alphas(t-1) )^2 + rho*( alphas(t+1) - alpha )^2;
	else
		cost = -splnApprx(alpha) + rho*( alpha - alphas(t-1) )^2;
	end
	
end

function [lambdaInit] = admm_optimization( lambdaInit, rho, splnApprx, lb, ub)

	[T] = numel(lambdaInit);
	opti_settings = optimset('Display','off','MaxFunEvals',1e10,'Maxiter',1e10,'TolX',1e-10,'TolFun',1e-10,'LargeScale','off');
	
	lambdaPrev = lambdaInit;
	epsilon = 1e-1;
	
	kk = 1;
	while true

		% min lambdas
		for tt = 1:T
			objFcn = @(alpha) separateObjective( alpha, tt, lambdaInit, rho/T, splnApprx{tt});
			[alpha, fval, extraflag] = fmincon( objFcn,  lambdaInit(tt), [],[],[],[],lb(tt),ub(tt) ,[],opti_settings);
			lambdaInit(tt) = alpha;
		end
		
		stepSize = norm( lambdaPrev - lambdaInit ,2);
		fval = jointObjective( lambdaInit, rho, splnApprx);
		
		fprintf('Iter: %d. Current cost: %0.3f. Step size: %0.3f\n',kk,fval,stepSize);
		if stepSize < epsilon
			break;
		end
		
		lambdaPrev = lambdaInit;
		kk = kk +1;
		
	end
end

