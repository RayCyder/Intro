function train_main()
%TRAIN_MAIN  Main training entry (clean orchestration only)

%% ========= Environment / Path / Seed =========
utils.ensureOnPath();   % add project root so +packages resolve
utils.setSeed(888);     % deterministic training runs

useGPU = canUseGPU();   % decide device once; keep loop clean

%% ========= Logging (startup) =========
logMsg('train_main start');
logMsg(sprintf('MATLAB: %s | OS: %s', version, computer));
if useGPU
    try
        g = gpuDevice;
        logMsg(sprintf('Device: GPU | %s | CC %s | %.1f GB', g.Name, g.ComputeCapability, g.TotalMemory/1e9));
    catch
        logMsg('Device: GPU | gpuDevice() info unavailable');
    end
else
    logMsg('Device: CPU');
end

%% ========= Config =========
cfg.dataRoot    = fullfile(pwd, 'cifar10');      % CIFAR-10 cache directory (will download/extract .mat batches here)
cfg.numClasses  = 10;
cfg.inputSize   = [32 32 3];

cfg.batchSize   = 128;
cfg.numEpochs   = 100;

cfg.baseLR      = 0.05/4;                        % align with python baseline
cfg.minLR       = cfg.baseLR * 0.1;
cfg.weightDecay = 5e-4;

cfg.precision   = 'single';                      % keep numeric types consistent
cfg.printEvery  = 50;

% Optimizer selection: 'adamw' or 'mvr2'
cfg.optimizer   = 'adamw';

% MVR2 (Muon) hyperparams
cfg.mvr2.mu       = 0.95;
cfg.mvr2.gamma    = 0.025;
cfg.mvr2.eps      = 1e-8;
cfg.mvr2.nSteps   = 3;
cfg.mvr2.isApprox = true;  % true: approx (faster); false: exact (extra backward)

logMsg('Config summary');
logMsg(sprintf('  dataRoot=%s', cfg.dataRoot));
logMsg(sprintf('  inputSize=[%d %d %d], numClasses=%d', cfg.inputSize(1), cfg.inputSize(2), cfg.inputSize(3), cfg.numClasses));
logMsg(sprintf('  epochs=%d, batchSize=%d, baseLR=%.4g, minLR=%.4g, wd=%.4g, precision=%s', ...
    cfg.numEpochs, cfg.batchSize, cfg.baseLR, cfg.minLR, cfg.weightDecay, cfg.precision));
logMsg(sprintf('  optimizer=%s', cfg.optimizer));
if strcmpi(cfg.optimizer,'mvr2')
    logMsg(sprintf('  mvr2: mu=%.4g, gamma=%.4g, eps=%.4g, nSteps=%d, isApprox=%d', ...
        cfg.mvr2.mu, cfg.mvr2.gamma, cfg.mvr2.eps, cfg.mvr2.nSteps, cfg.mvr2.isApprox));
end

%% ========= Data =========
logMsg('Preparing CIFAR-10 datastores...');
[dsTrain, dsVal, dsTest, classNames] = data.makeCIFAR10Datastores(cfg.dataRoot, 'ValRatio', 0.1, 'Seed', 888);

nTrain = safeNumObs(dsTrain);
nVal   = safeNumObs(dsVal);
nTest  = safeNumObs(dsTest);
if nTrain >= 0
    logMsg(sprintf('Dataset sizes: train=%d, val=%d, test=%d', nTrain, nVal, nTest));
else
    logMsg('Dataset sizes: (NumObservations unavailable)');
end
logMsg(sprintf('Classes (%d): %s', numel(classNames), strjoin(classNames(:).', ', ')));

mbqTrain = minibatchqueue(dsTrain, ...
    'MiniBatchSize', cfg.batchSize, ...
    'MiniBatchFcn', @(X,Y) preprocessMiniBatchAny(X,Y,classNames,cfg), ...
    'MiniBatchFormat', {'SSCB','CB'}, ...
    'PartialMiniBatch','discard');

mbqVal = minibatchqueue(dsVal, ...
    'MiniBatchSize', cfg.batchSize, ...
    'MiniBatchFcn', @(X,Y) preprocessMiniBatchAny(X,Y,classNames,cfg), ...
    'MiniBatchFormat', {'SSCB','CB'}, ...
    'PartialMiniBatch','discard');

mbqTest = minibatchqueue(dsTest, ...
    'MiniBatchSize', cfg.batchSize, ...
    'MiniBatchFcn', @(X,Y) preprocessMiniBatchAny(X,Y,classNames,cfg), ...
    'MiniBatchFormat', {'SSCB','CB'}, ...
    'PartialMiniBatch','discard');

logMsg('minibatchqueue created: Train/Val/Test');
logMsg(sprintf('  MiniBatchSize=%d | Formats={%s,%s}', cfg.batchSize, 'SSCB', 'CB'));

%% ========= Model =========
lgraph = models.resnet18_cifar10_layerGraph(cfg.numClasses, cfg.inputSize);
net = dlnetwork(lgraph);
net = dlupdate(@(p) cast(p,cfg.precision), net); % align parameters to precision

logMsg('Model created');
try
    L = net.Learnables;
    logMsg(sprintf('  Learnables: %d params tensors', height(L)));
catch
    logMsg('  Learnables: (unavailable)');
end

%% ========= Optimizer Init =========
switch lower(cfg.optimizer)
    case 'adamw'
        opt = optimizers.adamw.init(net, cfg.baseLR, cfg.weightDecay);
    case 'mvr2'
        opt = optimizers.muonmvr2.init(net, ...
            'LR', cfg.baseLR, ...
            'WeightDecay', cfg.weightDecay, ...
            'Mu', cfg.mvr2.mu, ...
            'Gamma', cfg.mvr2.gamma, ...
            'Eps', cfg.mvr2.eps, ...
            'NSteps', cfg.mvr2.nSteps, ...
            'IsApprox', cfg.mvr2.isApprox);
    otherwise
        error('Unknown optimizer: %s', cfg.optimizer);
end

logMsg('Optimizer initialized');
try
    logMsg(sprintf('  Initial LR=%.4g', opt.LR));
catch
end

%% ========= Train =========
iteration = 0;
printedFirstBatch = false;          % print shapes once for quick debugging
printedFirstEpochBatch = false;     % print shapes at epoch 1 first iter

for epoch = 1:cfg.numEpochs
    reset(mbqTrain);

    printedFirstEpochBatch = false;
    logMsg(sprintf('Epoch %d/%d start', epoch, cfg.numEpochs));

% LR schedule (cosine per epoch)
    opt.LR = utils.cosineLR(epoch, cfg.numEpochs, cfg.baseLR, cfg.minLR);
    logMsg(sprintf('  LR set to %.4g', opt.LR));

    tEpoch = tic;
    runningLoss = 0;

    while hasdata(mbqTrain)
        iteration = iteration + 1;

        [X, T] = next(mbqTrain);   % X: dlarray('SSCB'), T: onehot dlarray('CB')
        if useGPU
            X = gpuArray(X); T = gpuArray(T);
        end

        % Print first batch info (once) to confirm shapes/formats/types.
        if ~printedFirstBatch
            printedFirstBatch = true;
            logMsg('First batch (global) info');
            logDlarray('  X', X);
            logDlarray('  T', T);
        end
        if epoch == 1 && ~printedFirstEpochBatch
            printedFirstEpochBatch = true;
            logMsg('First batch (epoch 1) info');
            logDlarray('  X', X);
            logDlarray('  T', T);
        end

        % exact MVR2 needs old net gradients on same batch
        if strcmpi(cfg.optimizer,'mvr2') && ~opt.IsApprox
            netOld = net;
        end

        % compute loss and grads
        try
            [loss, grads] = dlfeval(@train.modelGradients, net, X, T);
        catch ME
            logMsg('ERROR during modelGradients');
            logMsg(sprintf('  message: %s', ME.message));
            logDlarray('  X', X);
            logDlarray('  T', T);
            rethrow(ME);
        end

        % NaN/Inf guard
        lossVal = double(gather(extractdata(loss)));
        if ~isfinite(lossVal)
            logMsg(sprintf('WARNING: non-finite loss detected: %g', lossVal));
        end

        runningLoss = runningLoss + lossVal;

        % exact MVR2: compute grads on old parameters (same batch)
        if strcmpi(cfg.optimizer,'mvr2') && ~opt.IsApprox
            [~, gradsOld] = dlfeval(@train.modelGradients, netOld, X, T);
            opt = optimizers.muonmvr2.updateLastGrad(opt, gradsOld);
        end

        % update parameters with selected optimizer
        switch lower(cfg.optimizer)
            case 'adamw'
                [net, opt] = optimizers.adamw.step(net, grads, opt);
            case 'mvr2'
                [net, opt] = optimizers.muonmvr2.step(net, grads, opt);
        end

        % periodic logging
        if mod(iteration, cfg.printEvery) == 0
            avgLoss = runningLoss / cfg.printEvery;
            runningLoss = 0;
            gnorm = gradL2Norm(grads);
            fprintf('Epoch %3d/%3d | Iter %6d | LR %.4g | Loss %.4f | ||g|| %.3e\n', ...
                epoch, cfg.numEpochs, iteration, opt.LR, avgLoss, gnorm);
        end
    end

    % eval at epoch end (no weight updates)
    logMsg('Running evaluation (val/test)...');
    [valAcc, valLoss] = utils.evaluate(net, mbqVal, useGPU);
    [testAcc, testLoss] = utils.evaluate(net, mbqTest, useGPU);
    fprintf('==> Epoch %3d done (%.1fs) | Val Acc %.2f%% | Val Loss %.4f | Test Acc %.2f%% | Test Loss %.4f\n', ...
        epoch, toc(tEpoch), valAcc*100, valLoss, testAcc*100, testLoss);
    logMsg(sprintf('Epoch %d end', epoch));
end

function [X, T] = preprocessMiniBatchAny(X, Y, classNames, cfg)
    % Accept either cell-array images (imageDatastore pipeline) or 4-D arrays (arrayDatastore pipeline).
    if iscell(X)
        X = cat(4, X{:});
    end

    X = im2single(X);

    meanRGB = reshape(single([0.4914 0.4822 0.4465]), 1,1,3);
    stdRGB  = reshape(single([0.2023 0.1994 0.2010]),  1,1,3);
    X = (X - meanRGB) ./ stdRGB;

    X = dlarray(X, 'SSCB');

    % ---- Robust label handling for different minibatchqueue outputs ----
    % From arrayDatastore of categorical scalars, minibatchqueue may return Y as a cell array.
    if iscell(Y)
        if isempty(Y)
            Y = categorical([], classNames);
        elseif all(cellfun(@iscategorical, Y))
            % Cell array of scalar categoricals -> convert to cellstr of names
            Y = categorical(cellfun(@char, Y, 'UniformOutput', false), classNames);
        elseif all(cellfun(@isnumeric, Y))
            % Cell array of numeric scalars (0..9)
            Yn = cell2mat(Y);
            Y = categorical(classNames(double(Yn)+1), classNames);
        else
            % Assume cellstr / string-like
            Y = categorical(Y, classNames);
        end
    end

    % If still not categorical, convert.
    if ~iscategorical(Y)
        if isnumeric(Y)
            % CIFAR-10 numeric labels are typically 0..9
            Y = categorical(classNames(double(Y)+1), classNames);
        else
            Y = categorical(Y, classNames);
        end
    end

    % Ensure category set and order matches classNames without re-constructing categoricals.
    Y = setcats(Y, classNames);
    Y = reordercats(Y, classNames);

    if any(isundefined(Y))
        error('Found undefined labels in mini-batch. Check classNames alignment.');
    end

    % onehotencode expands along a singleton dimension. Y is typically Bx1, so expand along dim=2
    % to get BxK, then transpose to KxB for 'CB' format.
    Y = Y(:);  % ensure column
    T = onehotencode(Y, 2, 'ClassNames', classNames);  % BxK
    T = T.';   % KxB

    T = dlarray(single(T), 'CB');

    X = cast(X, cfg.precision);
    T = cast(T, cfg.precision);
end

function logMsg(msg)
%logMsg  Timestamped logger
fprintf('[%s] %s\n', datestr(now,'HH:MM:SS'), msg);
end

function logDlarray(prefix, A)
%logDlarray  Print compact info about dlarray/gpuArray/numeric
try
    cls = class(A);
    isGPU = isa(A,'gpuArray');
    if isa(A,'dlarray')
        try
            fmt = dims(A);
        catch
            fmt = '';
        end
        sz = size(A);
        logMsg(sprintf('%s: %s%s | size=%s | dims=%s', prefix, cls, ternary(isGPU,'(gpu)',''), mat2str(sz), fmt));
    else
        sz = size(A);
        logMsg(sprintf('%s: %s%s | size=%s', prefix, cls, ternary(isGPU,'(gpu)',''), mat2str(sz)));
    end
catch
    logMsg(sprintf('%s: (unable to inspect)', prefix));
end
end

function n = safeNumObs(ds)
%safeNumObs  Best-effort NumObservations for combined datastore
n = -1;
try
    uds = ds.UnderlyingDatastores;
    if iscell(uds) && ~isempty(uds)
        d0 = uds{1};
        if isprop(d0,'NumObservations')
            n = d0.NumObservations;
        end
    end
catch
end
end

function gnorm = gradL2Norm(grads)
%gradL2Norm  Global L2 norm of gradients in Learnables table
try
    s = 0;
    for i = 1:height(grads)
        g = grads.Value{i};
        if isempty(g)
            continue;
        end
        gd = extractdata(gather(g));
        s = s + sum(gd(:).^2);
    end
    gnorm = sqrt(s);
catch
    gnorm = NaN;
end
end

function out = ternary(cond, a, b)
if cond
    out = a;
else
    out = b;
end
end

end
