function opt = init(net, varargin)
%init  Initialize Muon/MVR2 optimizer state.
%
% Usage:
%   opt = optimizers.muonmvr2.init(net, 'LR', 1e-3, 'WeightDecay', 0, ...)
%
% This initializer creates per-parameter state buffers (momentum, lastGrad)
% aligned with net.Learnables.

% ---- Parse hyperparameters ----
p = inputParser;
addParameter(p, 'LR', 1e-3);
addParameter(p, 'WeightDecay', 0);
addParameter(p, 'Mu', 0.95);
addParameter(p, 'Gamma', 0.025);
addParameter(p, 'Eps', 1e-8);
addParameter(p, 'NSteps', 3);
addParameter(p, 'IsApprox', true);
parse(p, varargin{:});

% ---- Basic checks ----
if nargin < 1 || isempty(net) || ~isa(net, 'dlnetwork')
    error('muonmvr2:init:InvalidNet', 'First argument must be a dlnetwork.');
end

L = net.Learnables;
numParams = height(L);

% ---- Pack hyperparams ----
opt = struct();
opt.LR          = p.Results.LR;
opt.WeightDecay = p.Results.WeightDecay;
opt.Mu          = p.Results.Mu;
opt.Gamma       = p.Results.Gamma;
opt.Eps         = p.Results.Eps;
opt.NSteps      = p.Results.NSteps;
opt.IsApprox    = logical(p.Results.IsApprox);

% Bookkeeping
opt.Step = 0;
opt.ParamTable = L(:, {'Layer','Parameter'});

% ---- Per-parameter buffers ----
% Momentum buffer (same shape/type/device as the parameter)
opt.M = cell(numParams, 1);

% LastGrad buffer is used only when IsApprox=false (exact variant).
% We allocate as empty now; updateLastGrad() can fill it.
opt.LastGrad = cell(numParams, 1);

for i = 1:numParams
    w = L.Value{i};
    if isempty(w)
        opt.M{i} = [];
        opt.LastGrad{i} = [];
        continue;
    end

    % Create zeros buffer matching parameter type/device.
    % NOTE: keep dlarray type so later arithmetic matches net learnables.
    opt.M{i} = dlupdate(@(x) zeros(size(x), 'like', x), w);
    opt.LastGrad{i} = [];
end

end
