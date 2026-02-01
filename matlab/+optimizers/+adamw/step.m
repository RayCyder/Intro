function [net, opt] = step(net, grads, opt)
L = net.Learnables;

for i=1:height(L)
    W = L.Value{i};
    g = grads.Value{i};
    if isempty(g), continue; end

    s = opt.State{i};
    s.step = s.step + 1;

    % decoupled weight decay
    if opt.WeightDecay > 0
        W = W * (1 - opt.LR*opt.WeightDecay);
    end

    % Adam moments
    s.m = opt.Beta1*s.m + (1-opt.Beta1)*g;
    s.v = opt.Beta2*s.v + (1-opt.Beta2)*(g.^2);

    mhat = s.m / (1 - opt.Beta1^s.step);
    vhat = s.v / (1 - opt.Beta2^s.step);

    W = W - opt.LR * mhat ./ (sqrt(vhat) + opt.Eps);

    L.Value{i} = W;
    opt.State{i} = s;
end

net = setLearnablesValue(net, L);
end
