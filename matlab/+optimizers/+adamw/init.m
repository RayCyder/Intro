function opt = init(net, lr, wd)
opt.LR = lr;
opt.WeightDecay = wd;
opt.Beta1 = 0.9;
opt.Beta2 = 0.999;
opt.Eps = 1e-8;

L = net.Learnables;
opt.State = cell(height(L),1);
for i=1:height(L)
    W = L.Value{i};
    s.step = 0;
    s.m = zeros(size(W),'like',W);
    s.v = zeros(size(W),'like',W);
    opt.State{i} = s;
end
end
