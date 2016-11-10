%% HELP:
%
%			AUTHOR:
%					Jaume Coll-Font <jcollfont@gmail.com>
%

%% main function
function [xk,zk,rho,lamk] = inverseTikhonovTSVD_geometry_ADMM_proofOfConcept(A,R,ECG,lambda,initialx,rho,min_r,min_s,trunk, verbose,zk,lamk)

	% DEFINE
		[N, M] =size(A);
		[M,T,K] = size(initialx);
		
	% matrices for rapid computations	
		revisit_rho = 10000;
		
	% SET UP PROBLEM
		U = zeros(M,M,K);
		iS = zeros(M,K);
		V = zeros(M,M,K);
		c = zeros(M,T,K);
		Q = zeros(M,M,K);
		for ii = 1:K
			Q(:,:,ii) = squeeze(A(:,:,ii))'*squeeze(A(:,:,ii)) + lambda*R'*R;
			c(:,:,ii) = -2*squeeze(A(:,:,ii))'*ECG(:,:,ii);
			
			[U(:,:,ii), S, V(:,:,ii)] = svd( 2*Q(:,:,ii) + rho*eye(M) );
			iS(:,ii) = diag(S).^(-1);
		end
		
	% INITIALIZE with WARM START
		xk = initialx;
		
		if exist('zk') && exist('lamk') && ( numel(zk)*numel(lamk)~=0 )
			[zk] = min_L_z(xk,lamk,rho);
		else
			zk = mean(xk,3);
		end
		
		[rk,sk] = residuals(xk,zk,zk,rho);
		
		if ~exist('lamk') || numel(lamk)==0
			lamk = rho*rk;
		end
		
		k = 1;
		
	% ADMM
	while true
		
		% min f(x)
			[xk] = min_L_x_overdet(U,iS,V,c,xk,zk,lamk,rho,trunk);
			
		% min g(z)
			zk1 = zk;
			[zk] = min_L_z(xk,lamk,rho);

		% compute residuals
			[rk,sk] = residuals(xk,zk,zk1,rho);
			
		% min lam
			lamk = lamk + rho*rk;
			
		% primal and dual residual norms
			nrk = norm(rk(:),2);
			nsk = norm(sk(:),2);
			
		% verbose and stopping criteria
			if verbose; fprintf('Iter: %d. Primal residual: %0.6f. Dual residual %0.6f.\n',k,nrk,nsk);end
			k = k+1;
			if ( nrk < min_r )&&( nsk < min_s )
				if verbose;fprintf('GatoDominguez!\n');end
				return;
			end
			
		% update adaptive rho
			if mod(k,revisit_rho) == 0
				[rho, U,iS,V] = new_rho(nrk,nsk,rho,Q,U,iS,V);
			end
			
	end

end


%% min f(x) --- actual objective function (LSQ)
%		Optimize over the fitting error function. This is the Least Squares
%		problem.
%
function [xk] = min_L_x_overdet(U,iS,V,c,xk,zk,lamk,rho,trunk)

	[M,T,K] = size(xk);
	
	for ii = 1:K
% 		xk(:,:,ii) = zeros(M,T);
		lhs = ( -c(:,:,ii) + rho*zk - lamk(:,:,ii) );
% 		for ss = 1:trunk
% 			xk(:,:,ii) =  xk(:,:,ii) + S(ss,ss,ii).^(-1)*V(:,ss,ii)*U(:,ss,ii)'*lhs;
% 		end
% 		
		xk(:,:,ii) = ( V(:,1:trunk,ii)*diag(iS(1:trunk,ii))*U(:,1:trunk,ii)' )*lhs;
		
	end
	
end


%% min g(x) --- constraints
%	Optimizes over the constraint functions.
%
function [zk] = min_L_z(xk,lamk,rho)

	zk = mean(xk + lamk/rho ,3);
	
end


%% compute residuals
%	Computes the new residuals (primal and dual) at each iteration.
%
function [rk,sk] = residuals(xk,zk,zk1,rho)


	rk = xk - repmat(zk,[1,1,size(xk,3)]);
	
	sk = -rho*(zk1 - zk);
	
end

%% update rho
% every revisit_rho iterations checks the difference between residuals and
% changes rho appropriately.
%
%	If		r > mu*s -> rho = tau*rho;
%	elseif	s > mu*r -> rho = 1/tau*rho;
%
function [rho,U,iS,V] = new_rho(nrk,nsk,rho,Q,U,iS,V)
	
	mu = 10;
	tau = 2;
	M = size(Q,1);

	if (nrk > mu*nsk)
		rho = tau*rho;
		for ii = 1:size(U,3)
			[U(:,:,ii), S, V(:,:,ii)] = svd( 2*Q(:,:,ii) + rho*eye(M) );
			iS(:,ii) = ( diag(S).^(-1) );
		end
	elseif (nsk > mu*nrk)
		rho = rho/tau;
		for ii = 1:size(U,3)
			[U(:,:,ii), S, V(:,:,ii)] = svd( 2*Q(:,:,ii) + rho*eye(M) );
			iS(:,ii) = ( diag(S).^(-1) );
		end
	end
end