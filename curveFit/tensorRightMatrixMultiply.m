function outtensor = tensorRightMatrixMultiply(intensor,rightIndex,rightMatrix)
% Author: Burak Erem

blockmat=num2cell(intensor,[1,rightIndex+1]);				% extract the intensor of the appropriate coordinates
matsize=size(blockmat{1});									% size of the extracted intensor
cellsize=size(blockmat);									% number of selected intensors
blockvecmat=squeeze(blockmat); blockvecmat=blockvecmat(:);	% format and vectorize
vecmat=squeeze(cat(1,blockvecmat{:}));						% concatenate the intensors extracted

multvecmat=vecmat*rightMatrix;								% multiply vectorized intensors with right matrix

REblockvecmat=mat2cell(multvecmat,matsize(1)*ones(1,size(multvecmat,1)/matsize(1)),size(multvecmat,2));		% divide into cells containing each itnensor part output is [N,Ki]
multcellsize=cellsize(setdiff(1:numel(cellsize),[1,rightIndex+1]));											% select the not-selected intensors before the ith
REblockvecmat=reshape(REblockvecmat,[1,1,multcellsize]);													% reshape and put result into ith dim
outtensor=cell2mat(REblockvecmat);																			% create a matrix with dims NxTXK1xK2
multinds=1:numel(cellsize); multinds(2:rightIndex)=3:rightIndex+1; multinds(rightIndex+1)=2;				
outtensor=permute(outtensor,multinds);																		% reorder and put the intensor of interest into the second dim

end
% intensor are knot points
% right matrix is the B-spline
