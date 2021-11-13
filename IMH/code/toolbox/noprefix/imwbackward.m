function varargout = imwbackward(varargin)
% VL_IMWBACKWARD  Image backward warping
%   J = VL_IMWBACKWARD(I, X, Y) returns the values of image I at
%   locations X,Y. X and Y are real matrices of arbitrary but
%   identical dimensions. I is bilinearly interpolated between samples
%   and extended with NaNs to the whole real plane.
%
%   [J,JX,JY] = VL_IMWBACKWARD(...) returns the warped derivatives JX and
%   JY too.
%
%   By default, VL_IMWBACKWARD() assumes that the image I uses the standard
%   coordinate system. VL_IMWBACKWARD(XR,YR,I,X,Y) assumes instead that I
%   is defined on a rectangular grid specified by the vectors XR and
%   YR.
%
%   VL_IMWBACKWARD() is less general than the MATLAB native function
%   INTERP2(), but it is significantly faster.
%
%   See also: IMWFORWARD(), INTERP2(), VL_HELP().
[varargout{1:nargout}] = vl_imwbackward(varargin{:});
