function [xk,zk,rho,lamk] = solve_messnarz_ADMM(A,R,ECG,lambda,initialx,rho,min_r,min_s,margin,verbose,zk,lamk)

	% DEFINE
		[N, M] =size(A);
		
		revisit_rho = 1000;
		
	% SET UP PROBLEM
			Q = A'*A + lambda*(R')*R;
			c = -2*A'*ECG;
			
			[U] = chol(2*Q + 2*rho*eye(M));

	% INITIALIZE with WARM START
		xk = initialx;
	
		if exist('zk') && exist('lamk') && ( numel(zk)*numel(lamk)~=0 )
			[zk] = min_L_z(xk,zk,lamk,rho,margin);
		else
			zk = [xk(:,1), (xk(:,2:end) - xk(:,1:end-1)) , xk(:,end)];
		end
		
		[rk,sk] = residuals(xk,zk,zk,rho,margin);
		
		if ~exist('lamk') || numel(lamk)==0
			lamk = rho*rk;
		end
		
		k = 1;
		
	% ADMM
	while true
		
		% min f(x)
			[xk] = min_L_x_overdet(U,c,xk,zk,lamk,rho);
			
		% min g(z)
			zk1 = zk;
			[zk] = min_L_z(xk,zk,lamk,rho,margin);

		% compute residuals
			[rk,sk] = residuals(xk,zk,zk1,rho,margin);
			
		% min lam
			lamk = lamk + rho*rk;
			
		% primal and dual residual norms
			nrk = norm(rk,2);
			nsk = norm(sk,2);
			
		% verbose and stopping criteria
			if verbose; fprintf('Iter: %d. Primal residual: %0.6f. Dual residual %0.6f.\n',k,nrk,nsk);end
			k = k+1;
			if ( nrk < min_r )&&( nsk < min_s )
				if verbose;fprintf('GatoDominguez!\n');end
				return;
			end
			
		% update adaptive rho
			if mod(k,revisit_rho) == 0
				[rho, U] = new_rho(nrk,nsk,rho,Q,U);
			end
			
	end

end


%% min f(x) --- actual objective function (LSQ)
%		Optimize over the fitting error function. This is the Least Squares
%		problem.
%
function [xk] = min_L_x_overdet(R,c,xk,zk,lamk,rho)

	[M T] = size(xk);
	nDiv = 4;
	
	% solve for f(x_1)
	xk(:,1) = R\(R'\( -c(:,1) + rho*(zk(:,1) - zk(:,2) + xk(:,2)) - lamk(:,1) +lamk(:,2) ));
	
	% solve for f(x_T)
	xk(:,T) = R\(R'\( -c(:,T) + rho*(xk(:,T-1) + zk(:,T) + zk(:,T+1)) - lamk(:,T) - lamk(:,T+1) ));
	
	% solve for x_i i=[2:T-1] in nDiv blocks with i's sorted randomly.
	indxT = randperm(T-1);
	indxT = indxT(indxT~=1);
	for div = 1:nDiv
		indx = indxT;
		xk(:,indx) =  R\(R'\( -c(:,indx) + rho*( xk(:,indx-1) + xk(:,indx+1) + zk(:,indx) - zk(:,indx+1) ) - lamk(:,indx) +lamk(:,indx+1) ));
	end

end


%% min g(x) --- constraints
%	Optimizes over the constraint functions.
%
function [zk] = min_L_z(xk,zk,lamk,rho,margin)

	[M, T] = size(xk);
	
	% min_{z_1} g(z_1)
% 	zk(:,1) = xk(:,1) + 1/rho*lamk(:,1); 
	
	% min_{z_i} g(z_i) i = [2:T]
	for ii = 2:T
		zk(:,ii) = (xk(:,ii) - xk(:,ii-1)) + 1/rho*lamk(:,ii);
	end
	
	% min_{z_{T+1}} g(z_{T+1})
	zk(:,T+1) = xk(:,T) + 1/rho*lamk(:,T+1);
	
	
	% apply projections
	zk( zk(:,1)<margin(1) , 1) = margin(1);
	mask = (zk(:,2:T) < 0); mask = [false(M,1) mask false(M,1)];
	zk( mask )   = 0;
	zk(zk(:,T+1)>margin(2),T+1) = margin(2);
	
end


%% compute residuals
%	Computes the new residuals (primal and dual) at each iteration.
%
function [rk,sk] = residuals(xk,zk,zk1,rho,margin)

	[M T] = size(xk);
	rk = zeros(M,T+1);
	sk = zeros(M,T+1);
	
	
	%% compute primal residuals
		rk(:,1) = xk(:,1) - zk(:,1);
		rk(:,2:T) = xk(:,2:end) - xk(:,1:end-1) - zk(:,2:T);
		rk(:,T+1) = xk(:,T) - zk(:,T+1);

	%% compute dual residuals
		dzk = rho*(zk - zk1);
		sk(:,1:T) = (dzk(:,1:T) - dzk(:,2:T+1));
		sk(:,T+1) = dzk(:,T+1);

end

%% update rho
% every revisit_rho iterations checks the difference between residuals and
% changes rho appropriately.
%
%	If		r > mu*s -> rho = tau*rho;
%	elseif	s > mu*r -> rho = 1/tau*rho;
%
function [rho,U] = new_rho(nrk,nsk,rho,Q,U)
	
	mu = 10;
	tau = 2;
	M = size(Q,1);

	if (nrk > mu*nsk)
		rho = tau*rho;
		[U] = chol(2*Q + 2*rho*eye(M));
	elseif (nsk > mu*nrk)
		rho = rho/tau;
		[U] = chol(2*Q + 2*rho*eye(M));
	end
end