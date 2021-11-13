function varargout = lbp(varargin)
% VL_LBP  Local Binary Patterns
%   F = VL_LBP(IM, CELLSIZE) computes the Local Binary Pattern (LBP)
%   features for image I.
%
%   IM is divided in cells of size CELLSIZE. F is a three-dimensional
%   array containing one histograms of quantized LBP features per
%   cell. The witdh of F is FLOOR(WIDTH/CELLSIZE), where WIDTH is the
%   width of the image. The same for the height. The third dimension
%   is 58.
%
%   See also: <a href="matlab:vl_help('lbp')">LBP</a>, VL_LBPFLIPLR(),
%   VL_HELP().
[varargout{1:nargout}] = vl_lbp(varargin{:});
