function opt = init(~, varargin)
% Placeholder: implement Muon/MVR2 optimizer state initialization here.

p = inputParser;
addParameter(p, "LR", 1e-3);
addParameter(p, "WeightDecay", 0);
addParameter(p, "Mu", 0.95);
addParameter(p, "Gamma", 0.025);
addParameter(p, "Eps", 1e-8);
addParameter(p, "NSteps", 3);
addParameter(p, "IsApprox", true);
parse(p, varargin{:});

opt = p.Results;
opt.LastGrad = [];

error("muonmvr2:init:NotImplemented", "Muon/MVR2 optimizer is not implemented yet.");
end
