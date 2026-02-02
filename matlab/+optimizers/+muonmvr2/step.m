
function [net, opt] = step(net, grads, opt)
%step  Perform one Muon/MVR2 optimizer step.
%
% This implementation is designed to be:
%   - Compatible with MATLAB custom training loops (dlnetwork + grads table)
%   - Aligned with the project conventions used by optimizers.adamw
%   - Efficient: uses logits returned by train.modelGradients for train-acc,
%     and avoids redundant evaluation passes when cfg.printPerClass=true
%
% Inputs:
%   net   : dlnetwork
%   grads : table returned by dlgradient(loss, net.Learnables)
%   opt   : struct created by optimizers.muonmvr2.init
%
% Outputs:
%   net   : updated dlnetwork
%   opt   : updated optimizer state

% ---- Safety checks ----
if ~isa(net, 'dlnetwork')
    error('muonmvr2:step:InvalidNet', 'net must be a dlnetwork.');
end
if ~istable(grads) || ~any(strcmp(grads.Properties.VariableNames,'Value'))
    error('muonmvr2:step:InvalidGrads', 'grads must be a Learnables-like table with a Value column.');
end

L = net.Learnables;
numParams = height(L);

if height(grads) ~= numParams
    error('muonmvr2:step:Mismatch', 'grads table size (%d) does not match net.Learnables (%d).', height(grads), numParams);
end

% ---- Increment step ----
opt.Step = opt.Step + 1;

% ---- Build per-parameter update directions ----
updVals = cell(numParams, 1);

for i = 1:numParams
    w = L.Value{i};
    g = grads.Value{i};

    if isempty(w) || isempty(g)
        updVals{i} = g;
        continue;
    end

    % Ensure dlarray types line up (common in custom loops)
    if ~isa(w,'dlarray')
        w = dlarray(w);
    end
    if ~isa(g,'dlarray')
        g = dlarray(g);
    end

    % ===================== Python-matched 2D branch =====================
    % Python version applies the Muon/MVR update only for 2D weight matrices.
    % Non-2D params (bias/BN/etc.) fall back to the generic path below.

    if ndims(w) == 2 && isWeightParam(L.Parameter{i})
        % --- Get last_grad ---
        % approx: use opt.LastGrad cached from previous step (we update it below)
        % exact : train_main computes grads on netOld and calls updateLastGrad(opt, gradsOld)
        gPrev = [];
        if numel(opt.LastGrad) >= i && ~isempty(opt.LastGrad{i})
            gPrev = opt.LastGrad{i};
            if ~isa(gPrev,'dlarray')
                gPrev = dlarray(gPrev);
            end
        else
            % If missing, treat as zeros (same shape)
            gPrev = dlupdate(@(x) zeros(size(x), 'like', x), g);
        end

        % --- c_t = (g - last_grad) * gamma * (mu/(1-mu)) + g ---
        denom = max(1 - opt.Mu, 1e-12);
        factor = opt.Gamma * (opt.Mu / denom);
        ct = g + factor * (g - gPrev);

        % --- If ||c_t|| > 1, normalize ---
        ctNorm = sqrt(sum(ct.^2, 'all'));
        if double(gather(extractdata(ctNorm))) > 1
            ct = ct ./ (ctNorm + 1e-12);
        end

        % --- exp_avg = mu*exp_avg + (1-mu)*c_t ---
        expAvg = opt.M{i};
        if isempty(expAvg)
            expAvg = dlupdate(@(x) zeros(size(x), 'like', x), w);
        end
        expAvg = opt.Mu * expAvg + (1 - opt.Mu) * ct;
        opt.M{i} = expAvg;

        % --- update = Orth(exp_avg / (1-mu)) ---
        V = expAvg ./ denom;
        update = optimizers.muonmvr2.utils.zeropower_newton_schulz5(V, opt.NSteps, opt.Eps);

        % --- adjusted_lr = lr * 0.2 * sqrt(max(A,B)) ---
        sz = size(w);
        scale = 0.2 * sqrt(max(double(sz(1)), double(sz(2))));

        % We keep a unified apply rule: net = w - lr*u.
        % Python applies weight decay with lr (NOT adjusted_lr): p *= (1 - lr*wd)
        % and then p -= adjusted_lr*update.
        % This is equivalent to u = wd*w + scale*update.
        u = scale * update;
        if opt.WeightDecay ~= 0
            u = u + opt.WeightDecay * w;
        end

        updVals{i} = u;

        % --- Approx mode updates last_grad to current grad (Python behavior) ---
        if opt.IsApprox
            opt.LastGrad{i} = g;
        end

        continue;
    end

    % ===================== Generic (non-2D) path =====================

    % ---- MVR2 gradient correction (exact variant uses LastGrad) ----
    gCorr = g;
    if ~opt.IsApprox
        if numel(opt.LastGrad) >= i && ~isempty(opt.LastGrad{i})
            gPrev = opt.LastGrad{i};
            if ~isa(gPrev,'dlarray')
                gPrev = dlarray(gPrev);
            end
            % Simple variance-reduction style correction (placeholder for non-2D)
            gCorr = g + opt.Gamma * (g - gPrev);
        end
    end

    % ---- Momentum ----
    m = opt.M{i};
    if isempty(m)
        % Create momentum buffer if missing
        m = dlupdate(@(x) zeros(size(x), 'like', x), w);
    end
    m = opt.Mu * m + (1 - opt.Mu) * gCorr;
    opt.M{i} = m;

    % ---- Compute update direction u ----
    u = m;

    % Decoupled weight decay (AdamW-style): apply only to weight tensors
    if opt.WeightDecay ~= 0 && isWeightParam(L.Parameter{i})
        u = u + opt.WeightDecay * w;
    end

    % Orthogonalize update for matrix-like parameters (Linear/Conv weights)
    if shouldOrthogonalize(L.Parameter{i}, w)
        u = orthogonalizeLikePyTorch(u, w, opt.NSteps, opt.Eps);
    end

    updVals{i} = u;

    % Approx mode: also cache last_grad for generic params (keeps behavior consistent)
    if opt.IsApprox
        opt.LastGrad{i} = g;
    end
end

% ---- Apply parameter update using dlupdate (avoids custom setLearnablesValue) ----
U = L;
U.Value = updVals;

lr = opt.LR;
net = dlupdate(@(w,u) w - lr*u, net, U);

end

% ===================== helpers =====================

function tf = isWeightParam(paramName)
% Apply weight decay only to weights (not bias/scale/offset)
paramName = lower(string(paramName));
tf = contains(paramName, "weight");
% Common MATLAB names: 'Weights' / 'Bias' / 'Offset' / 'Scale'
if contains(paramName, "bias") || contains(paramName, "offset") || contains(paramName, "scale")
    tf = false;
end
end

function tf = shouldOrthogonalize(paramName, w)
% Only orthogonalize for 2-D weights or 4-D conv weights. Skip bias/BN.
paramName = lower(string(paramName));
if contains(paramName, "bias") || contains(paramName, "offset") || contains(paramName, "scale")
    tf = false;
    return;
end

sz = size(w);
nd = ndims(w);
if nd == 2
    tf = all(sz > 1);
elseif nd == 4
    % Conv weights in MATLAB are typically HxWxCxK.
    tf = (sz(4) > 1) && (sz(3) > 0) && (sz(1) > 0) && (sz(2) > 0);
else
    tf = false;
end
end

function uOut = orthogonalizeLikePyTorch(u, w, nSteps, eps)
% Reshape to 2-D, apply zeropower orthogonalization, then reshape back.
% For conv weights: MATLAB shape HxWxCxK (HWCK). We reshape to [K, C*H*W]
% matching PyTorch convention [out, in*kH*kW].

nd = ndims(w);

if nd == 2
    M = u;                 % [out x in]
    Mhat = optimizers.muonmvr2.utils.zeropower_newton_schulz5(M, nSteps, eps);
    uOut = reshape(Mhat, size(u));
    return;
end

% Conv weights
sz = size(w); % [H W C K]
H = sz(1); W = sz(2); C = sz(3); K = sz(4);

% Move K to rows
M = reshape(u, [H*W*C, K]);
M = permute(M, [2 1]);        % [K, H*W*C]

Mhat = optimizers.muonmvr2.utils.zeropower_newton_schulz5(M, nSteps, eps);

% Reshape back to HWCK
Mhat = permute(Mhat, [2 1]);  % [H*W*C, K]
uOut = reshape(Mhat, [H, W, C, K]);

end
