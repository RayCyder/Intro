function train_resnet18_cifar10_main()
% Main training pipeline for CIFAR-10 + ResNet18(CIFAR) using dlnetwork
% Requires: resnet18_cifar10_layerGraph.m in path

%% ============ Config ============
cfg.dataRoot = fullfile(pwd, "cifar10");   % 你需要准备: cifar10/train/<class>/*.png and cifar10/test/<class>/*.png
cfg.numClasses = 10;
cfg.inputSize  = [32 32 3];

cfg.batchSize  = 128;
cfg.numEpochs  = 100;

cfg.baseLR     = 0.05/4;  % 参考你 python 里写的 learning_rate = 0.05/4
cfg.minLR      = cfg.baseLR * 0.1;
cfg.weightDecay= 5e-4;

cfg.useGPU     = canUseGPU();
cfg.precision  = "single";  % 对齐 PyTorch float32
cfg.printEvery = 50;

% optimizer choice: "adamw" or "mvr2"
cfg.optimizer  = "mvr2";   % <-- 你要对齐 mvr2 就选这个；先跑通也可选 "adamw"

% MuonMVR2 params（按你脚本风格）
cfg.mvr2.mu     = 0.95;
cfg.mvr2.gamma  = 0.025;
cfg.mvr2.eps    = 1e-8;
cfg.mvr2.nSteps = 3;
cfg.mvr2.isApprox = true; % true=approx(更快)；false=exact(每步多一次反传)

%% ============ Data ============
[dsTrain, dsTest, classNames] = makeCIFAR10Datastores(cfg.dataRoot);

augTrain = augmentedImageDatastore(cfg.inputSize, dsTrain, ...
    "DataAugmentation", cifar10Augmenter(), ...
    "ColorPreprocessing", "none");

augTest  = augmentedImageDatastore(cfg.inputSize, dsTest, ...
    "ColorPreprocessing", "none");

mbqTrain = minibatchqueue(augTrain, ...
    "MiniBatchSize", cfg.batchSize, ...
    "MiniBatchFcn", @(X,Y) preprocessMiniBatch(X,Y,classNames,cfg), ...
    "MiniBatchFormat", {"SSCB","CB"}, ...
    "PartialMiniBatch", "discard");

mbqTest = minibatchqueue(augTest, ...
    "MiniBatchSize", cfg.batchSize, ...
    "MiniBatchFcn", @(X,Y) preprocessMiniBatch(X,Y,classNames,cfg), ...
    "MiniBatchFormat", {"SSCB","CB"}, ...
    "PartialMiniBatch", "discard");

%% ============ Model ============
lgraph = resnet18_cifar10_layerGraph(cfg.numClasses, cfg.inputSize);
net = dlnetwork(lgraph);

% (可选) 如果你要更接近 PyTorch 初始化，可在这里手动初始化（后续我也能给你一版“仿 PyTorch Kaiming-uniform”）
net = dlupdate(@(p) cast(p,cfg.precision), net);

%% ============ Optimizer init ============
switch lower(cfg.optimizer)
    case "adamw"
        opt = adamwInit(net, cfg.baseLR, cfg.weightDecay);
    case "mvr2"
        opt = muonMVR2Init(net, ...
            "LR", cfg.baseLR, ...
            "WeightDecay", cfg.weightDecay, ...
            "Mu", cfg.mvr2.mu, ...
            "Gamma", cfg.mvr2.gamma, ...
            "Eps", cfg.mvr2.eps, ...
            "NSteps", cfg.mvr2.nSteps, ...
            "IsApprox", cfg.mvr2.isApprox);
    otherwise
        error("Unknown optimizer: %s", cfg.optimizer);
end

%% ============ Train ============
iteration = 0;

for epoch = 1:cfg.numEpochs
    reset(mbqTrain);

    % 余弦退火学习率（与常见 PyTorch schedule 对齐）
    opt.LR = cosineLR(epoch, cfg.numEpochs, cfg.baseLR, cfg.minLR);

    tEpoch = tic;
    runningLoss = 0;

    while hasdata(mbqTrain)
        iteration = iteration + 1;
        [X, T] = next(mbqTrain); % X: dlarray('SSCB'), T: onehot dlarray('CB')

        if cfg.useGPU
            X = gpuArray(X);
            T = gpuArray(T);
        end

        % exact MVR2：需要旧网络在“同一 batch”上再算一次梯度
        if lower(cfg.optimizer) == "mvr2" && ~opt.IsApprox
            netOld = net;
        end

        % forward+backward
        [loss, grads] = dlfeval(@modelGradients, net, X, T);
        runningLoss = runningLoss + double(gather(extractdata(loss)));

        % exact MVR2: grads at old params on same batch
        if lower(cfg.optimizer) == "mvr2" && ~opt.IsApprox
            [~, gradsOld] = dlfeval(@modelGradients, netOld, X, T);
            opt = muonMVR2UpdateLastGrad(opt, gradsOld);
        end

        % step
        switch lower(cfg.optimizer)
            case "adamw"
                [net, opt] = adamwStep(net, grads, opt);
            case "mvr2"
                [net, opt] = muonMVR2Step(net, grads, opt);
        end

        % log
        if mod(iteration, cfg.printEvery) == 0
            avgLoss = runningLoss / cfg.printEvery;
            runningLoss = 0;
            fprintf("Epoch %3d/%3d | Iter %6d | LR %.4g | Loss %.4f\n", ...
                epoch, cfg.numEpochs, iteration, opt.LR, avgLoss);
        end
    end

    % Evaluate
    [testAcc, testLoss] = evaluate(net, mbqTest, cfg.useGPU);
    fprintf("==> Epoch %3d done (%.1fs) | Test Acc %.2f%% | Test Loss %.4f\n", ...
        epoch, toc(tEpoch), testAcc*100, testLoss);
end

end





function [dsTrain, dsTest, classNames] = makeCIFAR10Datastores(rootDir)
trainDir = fullfile(rootDir, "train");
testDir  = fullfile(rootDir, "test");

if ~isfolder(trainDir) || ~isfolder(testDir)
    error("Please prepare CIFAR-10 folders:\n  %s\n  %s\nEach must contain subfolders per class.", trainDir, testDir);
end

dsTrain = imageDatastore(trainDir, "IncludeSubfolders",true, "LabelSource","foldernames");
dsTest  = imageDatastore(testDir,  "IncludeSubfolders",true, "LabelSource","foldernames");

classNames = categories(dsTrain.Labels);
end


function augmenter = cifar10Augmenter()
% CIFAR typical: random translation (crop) + horizontal flip.
% We implement crop by random translation + fixed crop using output size fixed by augmentedImageDatastore.
augmenter = imageDataAugmenter( ...
    "RandXReflection", true, ...
    "RandXTranslation", [-4 4], ...
    "RandYTranslation", [-4 4]);
end


function [X, T] = preprocessMiniBatch(Xcell, Y, classNames, cfg)
% Xcell is a cell array of images
X = cat(4, Xcell{:});              % HxWxCxB
X = im2single(X);                  % float32

% PyTorch CIFAR常用 Normalize: mean=[0.4914 0.4822 0.4465], std=[0.2023 0.1994 0.2010]
% 你若 python 里用了别的 normalize，就把这里改成一致的即可。
meanRGB = reshape(single([0.4914 0.4822 0.4465]), 1,1,3);
stdRGB  = reshape(single([0.2023 0.1994 0.2010]),  1,1,3);
X = (X - meanRGB) ./ stdRGB;

X = dlarray(X, "SSCB");

% labels -> onehot (C,B)
Y = categorical(Y, classNames);
T = onehotencode(Y, 1, "ClassNames", classNames);  % CxB
T = dlarray(single(T), "CB");

% cast
X = cast(X, cfg.precision);
T = cast(T, cfg.precision);
end






function [loss, gradients] = modelGradients(net, X, T)
% net outputs logits (C,B) or (1,1,C,B) depending on layers
logits = forward(net, X);

% Make sure logits is CxB
if ndims(logits) == 4
    logits = squeeze(logits); % 1x1xC xB -> CxB
end

% softmax + cross entropy (equivalent to CE(logits) in PyTorch)
P = softmax(logits, "DataFormat","CB");
loss = -mean(sum(T .* log(P + 1e-12), 1));

gradients = dlgradient(loss, net.Learnables);
end





function [acc, avgLoss] = evaluate(net, mbq, useGPU)
reset(mbq);

numCorrect = 0;
numTotal = 0;
lossSum = 0;
numBatches = 0;

while hasdata(mbq)
    [X, T] = next(mbq);
    if useGPU
        X = gpuArray(X); T = gpuArray(T);
    end

    logits = forward(net, X);
    if ndims(logits) == 4
        logits = squeeze(logits);
    end
    P = softmax(logits, "DataFormat","CB");
    loss = -mean(sum(T .* log(P + 1e-12), 1));

    [~, pred] = max(extractdata(P), [], 1);
    [~, gt]   = max(extractdata(T), [], 1);

    numCorrect = numCorrect + sum(pred == gt);
    numTotal   = numTotal + numel(gt);

    lossSum = lossSum + double(gather(extractdata(loss)));
    numBatches = numBatches + 1;
end

acc = numCorrect / max(numTotal,1);
avgLoss = lossSum / max(numBatches,1);
end




function lr = cosineLR(epoch, numEpochs, baseLR, minLR)
t = (epoch-1) / max(numEpochs-1,1);
lr = minLR + 0.5*(baseLR - minLR)*(1 + cos(pi*t));
end






function opt = adamwInit(net, lr, wd)
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


function [net, opt] = adamwStep(net, grads, opt)
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