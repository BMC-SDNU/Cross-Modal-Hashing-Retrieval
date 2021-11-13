function [x, funVal, ValueL, res]=overlapping_LeastR(A, y, z, opts)
%
%%
% Function overlapping_LeastR
%      Least Squares Loss with the 
%           overlapping group Lasso
%
%% Problem
%
%  min  1/2 || A x - y||^2 + z_1 \|x\|_1 + z_2 * sum_i w_i ||x_{G_i}||
%
%  G_i's are nodes 
%
%    we have L1 for each element
%    and the L2 for the overlapping group 
%
%  The overlapping group information is contained in
% 
%   opts.G- a row vector containing the indices of all the overlapping
%           groups G_1, G_2, ..., G_groupNum
%    
%   opts.ind- a 3 x groupNum matrix
%           opts.ind(1,i): the starting index of G_i in opts.G
%           opts.ind(2,i): the ending index of G_i in opts.G
%           opts.ind(3,i): the weight for the i-th group
%
% For better illustration, we consider the following example of four groups:
%   G_1={1,2,3}, G_2={2,4}, G_3={3,5}, G_4={1,5}. 
% Let us assume the weight for each group is 123.
% 
%   opts.G=[1,2,3,2,4,3,5,1,5];
%   opts.ind=[ [1, 3, 123]', [4,5,123]',[6,7,123]',[8,9,123]' ];
%
%
%% Input parameters:
%
%  A-         Matrix of size m x n
%                A can be a dense matrix
%                         a sparse matrix
%                         or a DCT matrix
%  y -        Response vector (of size mx1)
%  z -        The regularization parameter (z=[z_1,z_2] >=0)
%  opts-      Optimal inputs (default value: opts=[])
%
%% Output parameters:
%
%  x-         Solution
%  funVal-    Function value during iterations
%
%% Copyright (C) 2010-2011 Jun Liu, and Jieping Ye
%
% You are suggested to first read the Manual.
%
% For any problem, please contact with Jun Liu via j.liu@asu.edu
%
% Last modified on April 21, 2010.
%
%% Related papers
%
% [1]  Jun Liu and Jieping Ye, Fast Overlapping Group Lasso, 
%      arXiv:1009.0306v1, 2010
%
%% Related functions:
%
%  sll_opts, initFactor, pathSolutionLeast
%
%%

%% Verify and initialize the parameters
%%
if (nargin <4)
    error('\n Inputs: A, y, z, and opts (.ind) should be specified!\n');
end

[m,n]=size(A);

if (length(y) ~=m)
    error('\n Check the length of y!\n');
end

if (z(1)<0 || z(2)<0)
    error('\n z should be nonnegative!\n');
end

lambda1=z(1);
lambda2=z(2);

opts=sll_opts(opts); % run sll_opts to set default values (flags)

% restart the program for better efficiency
%  this is a newly added function
if (~isfield(opts,'rStartNum'))
    opts.rStartNum=opts.maxIter;
else
    if (opts.rStartNum<=0)
        opts.rStartNum=opts.maxIter;
    end
end

%% Detailed initialization
%% Normalization

% Please refer to sll_opts for the definitions of mu, nu and nFlag
%
% If .nFlag =1, the input matrix A is normalized to
%                     A= ( A- repmat(mu, m,1) ) * diag(nu)^{-1}
%
% If .nFlag =2, the input matrix A is normalized to
%                     A= diag(nu)^{-1} * ( A- repmat(mu, m,1) )
%
% Such normalization is done implicitly
%     This implicit normalization is suggested for the sparse matrix
%                                    but not for the dense matrix
%

if (opts.nFlag~=0)
    if (isfield(opts,'mu'))
        mu=opts.mu;
        if(size(mu,2)~=n)
            error('\n Check the input .mu');
        end
    else
        mu=mean(A,1);
    end
    
    if (opts.nFlag==1)
        if (isfield(opts,'nu'))
            nu=opts.nu;
            if(size(nu,1)~=n)
                error('\n Check the input .nu!');
            end
        else
            nu=(sum(A.^2,1)/m).^(0.5); nu=nu';
        end
    else % .nFlag=2
        if (isfield(opts,'nu'))
            nu=opts.nu;
            if(size(nu,1)~=m)
                error('\n Check the input .nu!');
            end
        else
            nu=(sum(A.^2,2)/n).^(0.5);
        end
    end
    
    ind_zero=find(abs(nu)<= 1e-10);    nu(ind_zero)=1;
    % If some values in nu is typically small, it might be that,
    % the entries in a given row or column in A are all close to zero.
    % For numerical stability, we set the corresponding value to 1.
end

if (~issparse(A)) && (opts.nFlag~=0)
    fprintf('\n -----------------------------------------------------');
    fprintf('\n The data is not sparse or not stored in sparse format');
    fprintf('\n The code still works.');
    fprintf('\n But we suggest you to normalize the data directly,');
    fprintf('\n for achieving better efficiency.');
    fprintf('\n -----------------------------------------------------');
end

%% Group & Others 

% Initialize maxIter2
if (~isfield(opts,'maxIter2'))
    maxIter2=1000;
else
    maxIter2=opts.maxIter2;   
end
% the maximal number of iteration for the projection


% Initialize tol2
if (~isfield(opts,'tol2'))
    tol2=1e-8;
else
    tol2=opts.tol2;   
end
% the duality gap of the projection


% Initialize flag2
if (~isfield(opts,'flag2'))
    flag2=2;
else
    flag2=opts.flag2;   
end


% Initialize G 
if (~isfield(opts,'G'))
    error('\n In overlapping_LeastR, the field .G should be specified');
else
    G=opts.G-1;   
    % we substract 1, as in C, the index begins with 0
end

% Initialize w 
if (~isfield(opts,'ind'))
    error('\n In overlapping_LeastR, the field .ind should be specified');
else
    w=opts.ind;
    
    if (size(w,1)~=3)
        error('\n w is a 3 x groupNum matrix');
    end
    
    w(1:2,:)=w(1:2,:)-1;
    % we substract 1, as in C, the index begins with 0    
end

groupNum=size(w,2);
% the number of groups

Y=zeros(length(G),1);
% the starting point for the projection


%% Starting point initialization

% compute AT y
if (opts.nFlag==0)
    ATy =A'*y;
elseif (opts.nFlag==1)
    ATy= A'*y - sum(y) * mu';  ATy=ATy./nu;
else
    invNu=y./nu;              ATy=A'*invNu-sum(invNu)*mu';
end

% process the regularization parameter
if (opts.rFlag~=0)
    % z here is the scaling factor lying in [0,1]
%     if (lambda1<0 || lambda1>1 || lambda2<0 || lambda2>1)
%         error('\n opts.rFlag=1, and z should be in [0,1]');
%     end
    
    % compute lambda1_max
    temp=abs(ATy);
    lambda1_max=max(temp);
    
    lambda1=lambda1*lambda1_max;
    
    % compute lambda2_max(lambda_1)
    
    % lambda_2_max to be added later
    
    lambda2_max=1;
    
    lambda2=lambda2*lambda2_max;
end


% initialize a starting point
if opts.init==2
    x=zeros(n,1);
else
    if isfield(opts,'x0')
        x=opts.x0;
        if (length(x)~=n)
            error('\n Check the input .x0');
        end
    else
        x=ATy;  % if .x0 is not specified, we use ratio*ATy,
        % where ratio is a positive value
    end
end

% compute A x
if (opts.nFlag==0)
    Ax=A* x;
elseif (opts.nFlag==1)
    invNu=x./nu; mu_invNu=mu * invNu;
    Ax=A*invNu -repmat(mu_invNu, m, 1);
else
    Ax=A*x-repmat(mu*x, m, 1);     Ax=Ax./nu;
end

if (opts.init==0) 
    % ------  This function is not available
    %
    % If .init=0, we set x=ratio*x by "initFactor"
    % Please refer to the function initFactor for detail
    %
    
    % Here, we only support starting from zero, due to the complex
    % structure
    
    %x=zeros(n,1);    
end

%% The main program
% The Armijo Goldstein line search schemes + accelearted gradient descent

bFlag=0; % this flag tests whether the gradient step only changes a little

if (opts.mFlag==0 && opts.lFlag==0)    
    L=1;
    % We assume that the maximum eigenvalue of A'A is over 1
    
    % assign xp with x, and Axp with Ax
    xp=x; Axp=Ax; xxp=zeros(n,1);
    
    alphap=0; alpha=1;
    
    for iterStep=1:opts.maxIter
        % --------------------------- step 1 ---------------------------
        % compute search point s based on xp and x (with beta)
        beta=(alphap-1)/alpha;    s=x + beta* xxp;
        
        % --------------------------- step 2 ---------------------------
        % line search for L and compute the new approximate solution x
        
        % compute the gradient (g) at s
        As=Ax + beta* (Ax-Axp);
        
        % compute AT As
        if (opts.nFlag==0)
            ATAs=A'*As;
        elseif (opts.nFlag==1)
            ATAs=A'*As - sum(As) * mu';  ATAs=ATAs./nu;
        else
            invNu=As./nu;                ATAs=A'*invNu-sum(invNu)*mu';
        end
        
        % obtain the gradient g
        g=ATAs-ATy;
        
        % copy x and Ax to xp and Axp
        xp=x;    Axp=Ax;
        
        firstFlag=1;        
        while (1)
            % let s walk in a step in the antigradient of s to get v
            % and then do the L1/Lq-norm regularized projection
            v=s-g/L;
            
            % projection
            [x,gap,penalty2]=overlapping(v,  n, groupNum, lambda1/L, lambda2/L,...
                w, G, Y, maxIter2, flag2, tol2);
            
            
            if (nargout ==4)
                % record the number of iterations
                
                if (firstFlag)
                    res.projStep(iterStep)=penalty2(4);
                else
                    res.projStep(iterStep)=...
                        max(res.projStep(iterStep),penalty2(4) );
                end
            end            
            firstFlag=0;
            
            v=x-s;  % the difference between the new approximate solution x
            % and the search point s
            
            % compute A x
            if (opts.nFlag==0)
                Ax=A* x;
            elseif (opts.nFlag==1)
                invNu=x./nu; mu_invNu=mu * invNu;
                Ax=A*invNu -repmat(mu_invNu, m, 1);
            else
                Ax=A*x-repmat(mu*x, m, 1);     Ax=Ax./nu;
            end
            
            Av=Ax -As;
            r_sum=v'*v; l_sum=Av'*Av;
            
            if (r_sum <=1e-20)
                bFlag=1; % this shows that, the gradient step makes little improvement
                break;
            end
            
            % the condition is ||Av||_2^2 <= L * ||v||_2^2
            if(l_sum <= r_sum * L)
                break;
            else
                L=max(2*L, l_sum/r_sum);
                %fprintf('\n L=%5.6f',L);
            end
        end
        
        % --------------------------- step 3 ---------------------------
        % update alpha and alphap, and check whether converge
        alphap=alpha; alpha= (1+ sqrt(4*alpha*alpha +1))/2;
        
        xxp=x-xp;   Axy=Ax-y;
        
        ValueL(iterStep)=L;
        
        
        % function value = loss + regularizatioin
        funVal(iterStep)=Axy'* Axy/2 + lambda1 * sum(abs(x)) + lambda2 *penalty2 (1);
        
        if (nargout ==4)
            % record pp and gg;
            res.pp(iterStep)=penalty2 (2);
            res.qq(iterStep)=penalty2 (3);
            
            % record gap
            res.gap(iterStep)=gap;
            
            % record the number of zero group in the solution
            res.zg(iterStep)=penalty2(5);
        end
        
        if (bFlag)
            % fprintf('\n The program terminates as the gradient step changes the solution very small.');
            break;
        end
        
        switch(opts.tFlag)
            case 0
                if iterStep>=2
                    if (abs( funVal(iterStep) - funVal(iterStep-1) ) <= opts.tol)
                        break;
                    end
                end
            case 1
                if iterStep>=2
                    if (abs( funVal(iterStep) - funVal(iterStep-1) ) <=...
                            opts.tol* funVal(iterStep-1))
                        break;
                    end
                end
            case 2
                if ( funVal(iterStep)<= opts.tol)
                    break;
                end
            case 3
                norm_xxp=sqrt(xxp'*xxp);
                if ( norm_xxp <=opts.tol)
                    break;
                end
            case 4
                norm_xp=sqrt(xp'*xp);    norm_xxp=sqrt(xxp'*xxp);
                if ( norm_xxp <=opts.tol * max(norm_xp,1))
                    break;
                end
            case 5
                if iterStep>=opts.maxIter
                    break;
                end
        end
        
        % restart the program every opts.rStartNum
        if (~mod(iterStep, opts.rStartNum))
            alphap=0; alpha=1;
            xp=x; Axp=Ax; xxp=zeros(n,1); L =L/2;
        end
    end
else
    error('\n The function does not support opts.mFlag neq 0 & opts.lFlag neq 0!');
end