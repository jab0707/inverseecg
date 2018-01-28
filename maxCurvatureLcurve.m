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
%			- maxKappa - double - maximum curvature.
%
%		AUTHOR:
%			Jaume Coll-Font <jcollfont@gmail.com>
%
%


function [maxCurvatureIX, kappa, splnApp, maxKappa] = maxCurvatureLcurve(Gamma, lambdas, K)

	%% define
		[d] = size(Gamma,1);
		
		doplots = false;	% unless desired do not do plots
		
		% optimization settings
% 		opti_settings = optimset('Display','iter','MaxFunEvals',1e10,'Maxiter',1e10,'TolX',1e-10,'TolFun',1e-10,'LargeScale','off');
		
	if doplots
			plot(Gamma(1,:), Gamma(2,:),'o-');
			grid on;
	end
		
	%% clean out loops in the L-curve
		selIX = true(1,size(Gamma,2));
% 		[selIX] = selectValidLambdas(Gamma);
% 		Gamma = Gamma(:,selIX);
% 		lambdas = lambdas(selIX);
	
	%% fit spline approximation
		% create spline
			[Bspline, dS, ddS] = BsplineMatrix(K,lambdas);
		
		% LSQ fit
% 			objFcn = @(knots) norm( Gamma - knots*Bspline' ,'fro');
% 			knotInit = zeros(d,K);
% 			[knotOpt, fval, extraflag] = fminunc(objFcn,knotInit,opti_settings);
			knotOpt = Gamma*Bspline / ( Bspline'*Bspline);
		
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
		[maxKappa, ix] = max(kappa);
		temp = find(selIX);
		maxCurvatureIX = temp(ix);

		
	%% generate functions
% 		splnApp = @(t) (knotOpt*BsplineMatrix(K,t)');
		splnApp = @(t) computeCurvature(knotOpt, t, K, lambdas);
		
	%% plots
		if doplots
			hold on;
			plotSamp = splnApp(linspace(lambdas(1),lambdas(end),1000));
			plot(plotSamp(1,:),plotSamp(2,:),'g-','LineWidth',2);
			scatter(Gamma(1,:), Gamma(2,:),30,kappa,'fill');
			plot( Gamma(1,ix), Gamma(2,ix),'ro','LineWidth',2);
			plot(knotOpt(1,:),knotOpt(2,:),'kx','LineWidth',2);
			hold off;
			grid on;
			pause(0.2);
		end
		
end


%% create B-spline
function [S, dS, ddS] = BsplineMatrix(K,timeSamples)
	
	S = zeros(numel(timeSamples),K);
	

	for kk = 1:K
		
		% create B-spline functions
		temp = zeros(1,K);
		temp(kk)=1;
		
		startKnot = timeSamples(1) - 0*(timeSamples(end) - timeSamples(1))/K;
		endKnot = timeSamples(end) + 0*(timeSamples(end) - timeSamples(1))/K;
		
		ss=spline(linspace(startKnot,endKnot,K),temp);

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

function [kappa] = computeCurvature(knotOpt, t, K,lambdas)

		[~, dS, ddS] = BsplineMatrix(K,[lambdas(1), t, lambdas(end)]);
		
		dGamma = knotOpt*dS';
		ddGamma = knotOpt*ddS';
		
		kappa = ( dGamma(1,:).*ddGamma(2,:) - dGamma(2,:).*ddGamma(1,:) ) ./ ( sum(dGamma.^2,1) ).^(3/2);

		kappa = kappa(2:end-1);	

		
end

%% create spline
function [ssINT, dssINT, ddssINT] = splineInterp(K,Gamma,timeSamples)
	
	[D] = size(Gamma,1);
	T = numel(timeSamples);
	
	% create B-spline functions
	startKnot = timeSamples(1) - 0*(timeSamples(end) - timeSamples(1))/K;
	endKnot = timeSamples(end) + 0*(timeSamples(end) - timeSamples(1))/K;

	ssINT = zeros(D,T);
	dssINT = zeros(D,T);
	ddssINT = zeros(D,T);
	for dd = 1:D
		ss = spline(timeSamples, Gamma(dd,:));
		ssINT(dd,:) =ppval(ss,timeSamples);
		
		% first derivative function
		[breaks,coefs,l,k,d] = unmkpp(ss);
		dcoefs = zeros(l,k-1);
		for ll = 1:l
			dcoefs(ll,:) = polyder(coefs(ll,:));
		end
		dss = mkpp( breaks , dcoefs);
		dssINT(dd,:) =ppval(dss,timeSamples);

		% second derivative function
		[breaks,coefs,l,k,d] = unmkpp(dss);
		dcoefs = zeros(l,k-1);
		for ll = 1:l
			dcoefs(ll,:) = polyder(coefs(ll,:));
		end
		ddss = mkpp( breaks , dcoefs);

		ddssINT(dd,:) =ppval(ddss,timeSamples);

	end

end


%% fix the Lcurve from values that are too high or too low
function [selIX] = selectValidLambdas(Gamma)

	dGamma = (diff(Gamma,1,2));
	
	Nlam = size(Gamma,2);
	selIX = true(1,Nlam);
	
	[val, ix] = min(dGamma(1,:));
	range = max(Gamma(1,:)) - min(Gamma(1,:));
	
	if abs(val) > range/4
		selIX((ix+1):end) = false;
	else
		ix = Nlam;
	end
		
	for ii = (ix):-1:2
		
		[validIx] = checkNextLambda( Gamma(1,ii:-1:1), selIX(ii:-1:1));
		selIX(ii:-1:1) = validIx;
		
	end
	
end

function [validIx] = checkNextLambda( gamma, validIx)

	nextIx = find(validIx(2:end));
	
	if numel(nextIx)
		
		differ = gamma(nextIx(1)+1) - gamma(1);
	
		if differ < 0
			validIx((nextIx(1)+1):end) = false;
			if nextIx(1)+1 < numel(validIx)
				[validIx] = checkNextLambda( gamma, validIx);
			end
		else
			validIx(nextIx(1)+1) = true;
		end
	end
end