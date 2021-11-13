function varargout = ikmeanspush(varargin)
% VL_IKMEANSPUSH  Project data on integer K-means paritions
%   I = VL_IKMEANSPUSH(X,C) projects the data X to the integer K-meanns
%   clusters of centers C returning the cluster indeces I.
%
%   See also: VL_IKMEANS(), VL_HELP().
[varargout{1:nargout}] = vl_ikmeanspush(varargin{:});
