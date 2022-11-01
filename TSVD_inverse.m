function [EGM_sol, lambdaCornerIX,rho,eta] = TSVD_inverse(A, ECG, trunk, frobenius,manualTruncation)
    warning off
	% comopute svd of forward matrix
	[N,M] = size(A);
    
    [T] = size(ECG,2);
	
	% select maximum truncation
	maxTrunk = min(trunk, min(N,M));
	try
        [U,S,V] = svd(A);
    catch
        [U,S,V] = svds(A,max(N,M));
    end
	% project data onto left singular vectors
	Uy = U'*ECG;
	
	% for all dimensions of svd
	EGMtrunk = cell(1,maxTrunk);
	rho = zeros(maxTrunk, T);
	eta = zeros(maxTrunk, T);
	for k = 1:maxTrunk
		
		if k == 1
			EGMtrunk{1} = V(:,1) * 1/S(1,1) *Uy(1,:);
		else
			EGMtrunk{k} =  EGMtrunk{k-1} + V(:,k) * 1/S(k,k) * Uy(k,:);
		end
		
		rho(k,:) = sum( ( ECG - A*EGMtrunk{k}	).^2 , 1);
		eta(k,:) = sum( ( EGMtrunk{k}			).^2 , 1);
		
	end
	
	%% Choose to select independent lambda or single lambda
	EGM_sol = zeros(M,T);
    if manualTruncation == 0%IF manual truncation option is 0, then use the L curve, otehrwise use predefined truncation value
        if frobenius

            % compute L-curve and select corner
            rho = sum(rho,2)';
            eta = sum(eta,2)';

            [lambdaCornerIX, kappa] = maxCurvatureLcurve( log([rho;eta]), 1:maxTrunk, maxTrunk);

            % return solution
            EGM_sol = EGMtrunk{lambdaCornerIX};

        else

            % for every time instance
            for tt = 1:T

                % compute L-curve and select corner
                [lambdaCornerIX, kappa] = maxCurvatureLcurve(log([rho(:,tt)';eta(:,tt)']),  1:maxTrunk, maxTrunk);

                %% return solution
                EGM_sol(:,tt) = EGMtrunk{lambdaCornerIX}(:,tt);

            end

        end
    else
        EGM_sol = EGMtrunk{manualTruncation};%If a desired trunctation was provided use that
        lambdaCornerIX = manualTruncation;
        
    end
	
end