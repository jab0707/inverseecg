%% HELP:
%
%		This function approximates a given curve in 2D or 3D with a spline
%		function and then computes the curvature at each point (and picks
%		the max).
%
%		This function is sutited to compute the corner of an L-curve in the
%		inverse problems that need to resolve the regularization parameter.
%
%		The spline approximation is done with K knot points equally spaced
%		and then doing LSQ fit on the samples.
%		The curvature is computed as:
%
%		3D:		kappa = sqrt( (z''y' - y''z')^2 + (x''z' - z''x')^2 + (y''x' - x''y')^2 )
%									/( x'^2 + y'^2 +z'^2 )^(3/2);
%		2D:		kappa = (x'y'' - y'x'')
%							/( x'^2 + y'^2 )^(3/2);
%
%		INPUT:
%			- Gamma	 - <d,L>double - samples of the curve in 2D or 3D (d= 2 or 3).
%			- lambda - <1,L>double - parameter of the curve.
%			- K		 - int - (optional) number of knot points to use (default is 4).
%
%		OUTPUT:
%			- maxCurvatureIX - int - index along lambda with maximum curvature.
%			- kappa	 - <1,L>double - curvature at each sample along the curve.
%			%%- splnApp - fcn - spline approximation of the curve.
%			%%- kappaFcn - fcn - function describing the curvature along the curve.
%
%		AUTHOR:
%			Jaume Coll-Font <jcollfont@gmail.com>
%
%


function [maxCurvatureIX, kappa, splnApp] = maxCurvatureLcurve(Gamma, lambdas, K)

	%% define
		[d] = size(Gamma,1);
		
		% optimization settings
		opti_settings = optimset('Display','off','MaxFunEvals',1e10,'Maxiter',1e10,'TolX',1e-6,'TolFun',1e-6,'LargeScale','off');
	
	%% fit spline approximation
		% create spline
			[Bspline, dS, ddS] = BsplineMatrix(K,lambdas);
		
		% LSQ fit
			objFcn = @(knots) norm( Gamma - knots*Bspline' ,'fro');
			knotInit = zeros(d,K);
			
			[knotOpt, fval, extraflag] = fminunc(objFcn,knotInit,opti_settings);
		
	%% compute curvature and extract maximum
		dGamma = knotOpt*dS';
		ddGamma = knotOpt*ddS';
		
		if	   d == 2	% 2D
			
			kappa = ( dGamma(1,:).*ddGamma(2,:) - dGamma(2,:).*ddGamma(1,:) ) ./ ( sum(dGamma.^2,1) ).^(3/2);
		
		elseif d == 3	% 3D
		
			kappa = sqrt( (dGamma(2,:).*ddGamma(3,:) - dGamma(3,:).*ddGamma(2,:)).^2  + ...
						  (dGamma(3,:).*ddGamma(1,:) - dGamma(1,:).*ddGamma(3,:)).^2  + ...
						  (dGamma(1,:).*ddGamma(2,:) - dGamma(2,:).*ddGamma(1,:)).^2   )...
						    ./ ( sum(dGamma.^2,1) ).^(3/2);
			
		else
			kappa = [];
			
		end
	
		% maximum curvature
		[maxKappa, maxCurvatureIX] = max(kappa);

		
	%% generate functions
		splnApp = @(t) (knotOpt*BsplineMatrix(K,t)');
		
end


%% create B-spline
function [S, dS, ddS] = BsplineMatrix(K,timeSamples)
	
	S = zeros(numel(timeSamples),K);
	

	for kk = 1:K
		
		% create B-spline functions
		temp = zeros(1,K);
		temp(kk)=1;
		
		ss=spline(linspace(timeSamples(1),timeSamples(end),K),temp);

		% create B-spline matrix
		tempcol=ppval(ss,timeSamples);
		S(:,kk)=tempcol(:);
		
		% first derivative function
		[breaks,coefs,l,k,d] = unmkpp(ss);
		dcoefs = zeros(l,k-1);
		for ll = 1:l
			dcoefs(ll,:) = polyder(coefs(ll,:));
		end
		dss = mkpp( breaks , dcoefs);
% 		dss = fnder(ss);
		tempcol=ppval(dss,timeSamples);
		dS(:,kk)=tempcol(:);
		
		% second derivative function
		[breaks,coefs,l,k,d] = unmkpp(dss);
		dcoefs = zeros(l,k-1);
		for ll = 1:l
			dcoefs(ll,:) = polyder(coefs(ll,:));
		end
		ddss = mkpp( breaks , dcoefs);
		
		tempcol=ppval(ddss,timeSamples);
		ddS(:,kk)=tempcol(:);
		
	end
end