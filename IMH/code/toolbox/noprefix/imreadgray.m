function varargout = imreadgray(varargin)
% VL_IMREADGRAY  Reads an image as gray-scale
%   I=VL_IMREADGRAY(FILE) reads the image from file FILE and converts the
%   result to a gray scale image (DOUBLE storage class ranging in
%   [0,1]).
%
%   VL_IMREADGRAY(FILE,FMT) specifies the file format FMT (see IMREAD()).
%
%   See also: RGB2DOUBLE(), VL_HELP().
[varargout{1:nargout}] = vl_imreadgray(varargin{:});
