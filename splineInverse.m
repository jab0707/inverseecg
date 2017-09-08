function [EGM_sol] = splineInverse(A, ECG, RegMtrx, vec_lambda, underdetermined)

		
	%% PARAMETERS
	
	% curve interpolation
	NumberOfKnots=12;						% number of knots
	InterpolationDensity=100;				% interpolation density
	ProjectionInterpolationDensity=200;		% final projection density
	minderivcostreduction=500;				% minimum reconstruct err for deriv
	minoverallcostreduction= 1e-6;			% minimum reconstr err total
	
	
	%% FIT CURVE
		% Solve for initial derivatives
		fprintf('Fitting first and last curve derivatives...\n')
		CurveParams = initializeCurveParamsFromTimeSeries( ECG, NumberOfKnots);
		CurveParams = minimizeDistanceToCurve( CurveParams, ECG, InterpolationDensity, minderivcostreduction, 'JustDerivatives');

		% Solve for all parameters
		fprintf('Fitting all curve parameters...\n')
		CurveParams = minimizeDistanceToCurve( CurveParams, ECG, InterpolationDensity, minoverallcostreduction);

		% Project the data to the curve; obtain first time warp
		[Yproj, ProjInds] = ProjectDataPointsToCurve( CurveParams, ECG, ProjectionInterpolationDensity);
		Ywarp = InterpolateCurve( CurveParams, ProjectionInterpolationDensity);
						
	%% CALCULATE INVERSE
		fprintf('Computing Inverse...\n');
		[Xcurveparams] = tikhonov_jcf(A, RegMtrx, [], CurveParams, vec_lambda, underdetermined);
	
	%% RECONSTRUCT TIME SERIES
		fprintf('Reconstructing time signal...\n');
		Xwarp = InterpolateCurve(Xcurveparams,ProjectionInterpolationDensity);
		EGM_sol = Xwarp(:,ProjInds);
	
end