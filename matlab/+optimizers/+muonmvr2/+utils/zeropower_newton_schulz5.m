
function Y = zeropower_newton_schulz5(G, steps, eps)
%zeropower_newton_schulz5  Newton–Schulz (5th-order) orthogonalization / zeroth power.
%
% This is a MATLAB counterpart of the common PyTorch helper:
%   zeropower_via_newtonschulz5(G, steps=3, eps=1e-7)
%
% Given a 2-D matrix G (m x n), returns Y ≈ G (G^T G)^(-1/2), i.e., columns
% approximately orthonormal (semi-orthogonal). Used by Muon/MVR2-style
% optimizers to normalize/orthogonalize update directions.
%
% Inputs:
%   G     : numeric / gpuArray / dlarray, size [m n]
%   steps : number of Newton–Schulz iterations (default 3)
%   eps   : numerical epsilon (default 1e-7)
%
% Output:
%   Y     : same device/type as input (dlarray if input is dlarray)

if nargin < 2 || isempty(steps)
    steps = 3;
end
if nargin < 3 || isempty(eps)
    eps = 1e-7;
end

if isempty(G)
    Y = G;
    return;
end

% ---- Strip formatting / get raw numeric (keeps gpuArray if on GPU) ----
isDl = isa(G, 'dlarray');
if isDl
    Graw = extractdata(stripdims(G));
else
    Graw = G;
end

if ndims(Graw) ~= 2
    error('zeropower_newton_schulz5 expects a 2-D matrix.');
end

% Coefficients for a stable 5th-order iteration (used in several Muon/NS implementations)
a = 3.4445;
b = -4.7750;
c = 2.0315;

% ---- Normalize to help convergence ----
% Using Frobenius norm is cheap and keeps spectral radius typically < 1.
% Avoid divide-by-zero.
Gn2 = sum(Graw(:).*Graw(:));
Gn = sqrt(max(Gn2, 0));
scale = max(Gn, eps);
X = Graw ./ scale;

% ---- Newton–Schulz iterations ----
% X_{k+1} = X_k * (aI + bM + cM^2),  M = X_k^T X_k
for k = 1:steps
    M = X.' * X;                 % n x n
    M2 = M * M;

    % Identity on the same device/type as M
    I = eye(size(M,1), 'like', M);

    P = a*I + b*M + c*M2;
    X = X * P;
end

Yraw = X;

% ---- Return type ----
if isDl
    Y = dlarray(Yraw); % unformatted dlarray
else
    Y = Yraw;
end

end
