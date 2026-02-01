function train_main()
%TRAIN_MAIN  Main training entry (clean orchestration only)

%% ========= Environment / Path / Seed =========
utils.ensureOnPath();   % add project root so +packages resolve
utils.setSeed(888);     % deterministic training runs

useGPU = canUseGPU();   % decide device once; keep loop clean

%% ========= Config =========
cfg.dataRoot    = fullfile(pwd, "cifar10");      % CIFAR-10 root: train/ + test/ folders
cfg.numClasses  = 10;
cfg.inputSize   = [32 32 3];

cfg.batchSize   = 128;
cfg.numEpochs   = 100;

cfg.baseLR      = 0.05/4;                        % align with python baseline
cfg.minLR       = cfg.baseLR * 0.1;
cfg.weightDecay = 5e-4;

cfg.precision   = "single";                      % keep numeric types consistent
cfg.printEvery  = 50;

% Optimizer selection: "adamw" or "mvr2"
cfg.optimizer   = "adamw";

% MVR2 (Muon) hyperparams
cfg.mvr2.mu       = 0.95;
cfg.mvr2.gamma    = 0.025;
cfg.mvr2.eps      = 1e-8;
cfg.mvr2.nSteps   = 3;
cfg.mvr2.isApprox = true;  % true: approx (faster); false: exact (extra backward)

%% ========= Data =========
[dsTrain, dsTest, classNames] = data.makeCIFAR10Datastores(cfg.dataRoot);

augTrain = augmentedImageDatastore(cfg.inputSize, dsTrain, ...
    "DataAugmentation", data.cifar10Augmenter(), ...
    "ColorPreprocessing","none");

augTest  = augmentedImageDatastore(cfg.inputSize, dsTest, ...
    "ColorPreprocessing","none");

% minibatchqueue produces dlarray batches + one-hot labels
mbqTrain = minibatchqueue(augTrain, ...
    "MiniBatchSize", cfg.batchSize, ...
    "MiniBatchFcn", @(X,Y) data.preprocessMiniBatch(X,Y,classNames,cfg), ...
    "MiniBatchFormat", {"SSCB","CB"}, ...
    "PartialMiniBatch","discard");

mbqTest = minibatchqueue(augTest, ...
    "MiniBatchSize", cfg.batchSize, ...
    "MiniBatchFcn", @(X,Y) data.preprocessMiniBatch(X,Y,classNames,cfg), ...
    "MiniBatchFormat", {"SSCB","CB"}, ...
    "PartialMiniBatch","discard");

%% ========= Model =========
lgraph = models.resnet18_cifar10_layerGraph(cfg.numClasses, cfg.inputSize);
net = dlnetwork(lgraph);
net = dlupdate(@(p) cast(p,cfg.precision), net); % align parameters to precision

%% ========= Optimizer Init =========
switch lower(cfg.optimizer)
    case "adamw"
        opt = optimizers.adamw.init(net, cfg.baseLR, cfg.weightDecay);
    case "mvr2"
        opt = optimizers.muonmvr2.init(net, ...
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

%% ========= Train =========
iteration = 0;

for epoch = 1:cfg.numEpochs
    reset(mbqTrain);

% LR schedule (cosine per epoch)
    opt.LR = utils.cosineLR(epoch, cfg.numEpochs, cfg.baseLR, cfg.minLR);

    tEpoch = tic;
    runningLoss = 0;

    while hasdata(mbqTrain)
        iteration = iteration + 1;

        [X, T] = next(mbqTrain);   % X: dlarray('SSCB'), T: onehot dlarray('CB')
        if useGPU
            X = gpuArray(X); T = gpuArray(T);
        end

        % exact MVR2 needs old net gradients on same batch
        if strcmpi(cfg.optimizer,"mvr2") && ~opt.IsApprox
            netOld = net;
        end

        % compute loss and grads
        [loss, grads] = dlfeval(@train.modelGradients, net, X, T);

        runningLoss = runningLoss + double(gather(extractdata(loss)));

        % exact MVR2: compute grads on old parameters (same batch)
        if strcmpi(cfg.optimizer,"mvr2") && ~opt.IsApprox
            [~, gradsOld] = dlfeval(@train.modelGradients, netOld, X, T);
            opt = optimizers.muonmvr2.updateLastGrad(opt, gradsOld);
        end

        % update parameters with selected optimizer
        switch lower(cfg.optimizer)
            case "adamw"
                [net, opt] = optimizers.adamw.step(net, grads, opt);
            case "mvr2"
                [net, opt] = optimizers.muonmvr2.step(net, grads, opt);
        end

        % periodic logging
        if mod(iteration, cfg.printEvery) == 0
            avgLoss = runningLoss / cfg.printEvery;
            runningLoss = 0;
            fprintf("Epoch %3d/%3d | Iter %6d | LR %.4g | Loss %.4f\n", ...
                epoch, cfg.numEpochs, iteration, opt.LR, avgLoss);
        end
    end

    % eval at epoch end (no weight updates)
    [testAcc, testLoss] = utils.evaluate(net, mbqTest, useGPU);
    fprintf("==> Epoch %3d done (%.1fs) | Test Acc %.2f%% | Test Loss %.4f\n", ...
        epoch, toc(tEpoch), testAcc*100, testLoss);
end

end
