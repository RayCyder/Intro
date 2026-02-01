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

cfg.evalOnTest   = true;                       % whether to also evaluate on test each epoch
cfg.printPerClass = false;                     % per-class metrics on val/test (slower, prints a lot)
cfg.saveHistory  = true;                       % save losses/accs to .mat at end
cfg.plotHistory  = true;                       % plot curves at end

cfg.outDir       = fullfile(pwd,'runs');       % where to save history

% Quick debug subset (to validate the pipeline fast)
cfg.debugSubset = false;           % set true to use tiny datasets
cfg.debugTrainN = 100;
cfg.debugValN   = 10;
cfg.debugTestN  = 10;
cfg.debugSubsetMode = 'head';      % 'head' (take first N) or 'random'
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

% Apply tiny subset for quick debugging
if cfg.debugSubset
    cfg.numEpochs   = 10;               % override for quick runs
    nTr = safeNumObs(dsTrain);
    nVa = safeNumObs(dsVal);
    nTe = safeNumObs(dsTest);

    if strcmpi(cfg.debugSubsetMode,'random')
        utils.setSeed(888);
        idxTr = randperm(nTr, min(cfg.debugTrainN, nTr));
        idxVa = randperm(nVa, min(cfg.debugValN,   nVa));
        idxTe = randperm(nTe, min(cfg.debugTestN,  nTe));
    else
        idxTr = 1:min(cfg.debugTrainN, nTr);
        idxVa = 1:min(cfg.debugValN,   nVa);
        idxTe = 1:min(cfg.debugTestN,  nTe);
    end

    dsTrain = subset(dsTrain, idxTr);
    dsVal   = subset(dsVal,   idxVa);
    dsTest  = subset(dsTest,  idxTe);

    logMsg(sprintf('DEBUG subset enabled (%s): train=%d, val=%d, test=%d', ...
        cfg.debugSubsetMode, safeNumObs(dsTrain), safeNumObs(dsVal), safeNumObs(dsTest)));
end

% Training-only augmentation (PyTorch-aligned): RandomCrop(32,pad=4) + RandomHorizontalFlip
augTrain = data.cifar10Augmenter('OutputSize', [32 32], 'Padding', 4, 'FlipProb', 0.5);
logMsg('Augmentation: train=RandomCrop(pad=4)+HFlip(0.5), val/test=none');

mbqTrain = minibatchqueue(dsTrain, ...
    'MiniBatchSize', cfg.batchSize, ...
    'MiniBatchFcn', @(X,Y) data.preprocessMiniBatch(X,Y,classNames,cfg,'Augmenter',augTrain), ...
    'MiniBatchFormat', {'SSCB','CB'}, ...
    'PartialMiniBatch','discard');

mbqVal = minibatchqueue(dsVal, ...
    'MiniBatchSize', cfg.batchSize, ...
    'MiniBatchFcn', @(X,Y) data.preprocessMiniBatch(X,Y,classNames,cfg), ...
    'MiniBatchFormat', {'SSCB','CB'}, ...
    'PartialMiniBatch','discard');

mbqTest = minibatchqueue(dsTest, ...
    'MiniBatchSize', cfg.batchSize, ...
    'MiniBatchFcn', @(X,Y) data.preprocessMiniBatch(X,Y,classNames,cfg), ...
    'MiniBatchFormat', {'SSCB','CB'}, ...
    'PartialMiniBatch','discard');

logMsg('minibatchqueue created: Train/Val/Test');
logMsg(sprintf('  MiniBatchSize=%d | Formats={%s,%s} | TrainAug=ON', cfg.batchSize, 'SSCB', 'CB'));

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
% ---- History (epoch-level) ----
trainLossHist = zeros(cfg.numEpochs,1);
trainAccHist  = zeros(cfg.numEpochs,1);
valLossHist   = zeros(cfg.numEpochs,1);
valAccHist    = zeros(cfg.numEpochs,1);

if cfg.evalOnTest
    testLossHist  = zeros(cfg.numEpochs,1);
    testAccHist   = zeros(cfg.numEpochs,1);
else
    testLossHist  = [];
    testAccHist   = [];
end

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
    epochLossSum = 0;
    epochNumBatches = 0;
    epochCorrect = 0;
    epochTotal = 0;

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

        % compute loss, grads, and logits (for train accuracy)
        try
            [loss, grads, logits] = dlfeval(@train.modelGradients, net, X, T);
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
        epochLossSum = epochLossSum + lossVal;
        epochNumBatches = epochNumBatches + 1;

        % --- Train accuracy accumulation (printed once per epoch) ---
        % Use logits returned by train.modelGradients to avoid an extra forward pass.
        logitsAcc = logits;
        if ndims(logitsAcc) == 4
            logitsAcc = squeeze(logitsAcc);
        end
        isFormatted = false;
        if isa(logitsAcc,'dlarray')
            try
                isFormatted = ~isempty(dims(logitsAcc));
            catch
                isFormatted = false;
            end
        end
        if isFormatted
            Pacc = softmax(logitsAcc);
        else
            Pacc = softmax(logitsAcc, 'DataFormat', 'CB');
        end
        [~, predAcc] = max(gather(extractdata(Pacc)), [], 1);
        [~, gtAcc]   = max(gather(extractdata(T)),    [], 1);
        epochCorrect = epochCorrect + sum(predAcc == gtAcc);
        epochTotal   = epochTotal   + numel(gtAcc);

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

    % ---- Epoch summary (printed once per epoch) ----
    trainLoss = epochLossSum / max(epochNumBatches,1);
    trainAcc  = epochCorrect / max(epochTotal,1);

    logMsg('Running evaluation (val/test)...');

    % ---- Validation metrics (single pass) ----
    if cfg.printPerClass
        fprintf('Validation per-class metrics:\n');
        [valAcc, valLoss] = utils.evaluatePerClass(net, mbqVal, useGPU, classNames);
    else
        [valAcc, valLoss] = utils.evaluate(net, mbqVal, useGPU);
    end

    % ---- Test metrics (single pass, optional) ----
    if cfg.evalOnTest
        if cfg.printPerClass
            fprintf('Test per-class metrics:\n');
            [testAcc, testLoss] = utils.evaluatePerClass(net, mbqTest, useGPU, classNames);
        else
            [testAcc, testLoss] = utils.evaluate(net, mbqTest, useGPU);
        end
    else
        testAcc = NaN; testLoss = NaN;
    end

    % Record history for plotting
    trainLossHist(epoch) = trainLoss;
    trainAccHist(epoch)  = trainAcc;
    valLossHist(epoch)   = valLoss;
    valAccHist(epoch)    = valAcc;
    if cfg.evalOnTest
        testLossHist(epoch) = testLoss;
        testAccHist(epoch)  = testAcc;
    end

    % Python-like single-line summary
    if cfg.evalOnTest
        fprintf('[%3d] time: %.1fs train_loss: %.4f train_acc: %.2f%%  val_loss: %.4f val_acc: %.2f%%  test_loss: %.4f test_acc: %.2f%%  lr: %.6f\n', ...
            epoch, toc(tEpoch), trainLoss, trainAcc*100, valLoss, valAcc*100, testLoss, testAcc*100, opt.LR);
    else
        fprintf('[%3d] time: %.1fs train_loss: %.4f train_acc: %.2f%%  val_loss: %.4f val_acc: %.2f%%  lr: %.6f\n', ...
            epoch, toc(tEpoch), trainLoss, trainAcc*100, valLoss, valAcc*100, opt.LR);
    end
end

% ---- Save history + plot (optional) ----
history = struct();
history.trainLoss = trainLossHist;
history.trainAcc  = trainAccHist;
history.valLoss   = valLossHist;
history.valAcc    = valAccHist;
history.testLoss  = testLossHist;
history.testAcc   = testAccHist;
history.cfg       = cfg;

if cfg.saveHistory
    if ~exist(cfg.outDir, 'dir')
        mkdir(cfg.outDir);
    end
    stamp = datestr(now, 'yyyymmdd_HHMMSS');
    outPath = fullfile(cfg.outDir, ['run_' stamp '.mat']);
    save(outPath, 'history');
    logMsg(['Saved history to: ' outPath]);
end

if cfg.plotHistory
    figure('Name','Training Curves');
    subplot(2,1,1);
    plot(trainLossHist, '-'); hold on; plot(valLossHist, '-');
    if cfg.evalOnTest, plot(testLossHist, '-'); end
    grid on; xlabel('Epoch'); ylabel('Loss');
    if cfg.evalOnTest
        legend('train','val','test','Location','best');
    else
        legend('train','val','Location','best');
    end

    subplot(2,1,2);
    plot(trainAccHist*100, '-'); hold on; plot(valAccHist*100, '-');
    if cfg.evalOnTest, plot(testAccHist*100, '-'); end
    grid on; xlabel('Epoch'); ylabel('Accuracy (%)');
    if cfg.evalOnTest
        legend('train','val','test','Location','best');
    else
        legend('train','val','Location','best');
    end
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
%safeNumObs  Best-effort number of observations for a datastore.
%
% Different MATLAB versions expose different APIs/properties:
%   - ds.NumObservations
%   - numobservations(ds)
%   - combined datastore: ds.Datastores or ds.UnderlyingDatastores

n = -1;

% 1) Direct property (newer versions)
try
    if isprop(ds, 'NumObservations')
        n = ds.NumObservations;
        if ~isempty(n) && isnumeric(n) && isfinite(n) && n >= 0
            return;
        end
    end
catch
end

% 2) Function (newer versions)
try
    n2 = numobservations(ds);
    if ~isempty(n2) && isnumeric(n2) && isfinite(n2) && n2 >= 0
        n = n2;
        return;
    end
catch
end

% 3) Combined datastore: try to access first underlying datastore
uds = [];
try
    if isprop(ds, 'Datastores')
        uds = ds.Datastores;
    elseif isprop(ds, 'UnderlyingDatastores')
        uds = ds.UnderlyingDatastores;
    end
catch
end

try
    if ~isempty(uds)
        if iscell(uds)
            d0 = uds{1};
        else
            % some versions may store as a vector of datastores
            d0 = uds(1);
        end

        % Try the same strategies on the underlying datastore
        if isprop(d0, 'NumObservations')
            n = d0.NumObservations;
            if ~isempty(n) && isnumeric(n) && isfinite(n) && n >= 0
                return;
            end
        end

        try
            n2 = numobservations(d0);
            if ~isempty(n2) && isnumeric(n2) && isfinite(n2) && n2 >= 0
                n = n2;
                return;
            end
        catch
        end

        % Last resort for arrayDatastore: size of the underlying data
        try
            if isprop(d0, 'UnderlyingDatastore')
                d0 = d0.UnderlyingDatastore;
            end
        catch
        end

        try
            if isprop(d0, 'Data')
                n = size(d0.Data, 1);
                if isnumeric(n) && isfinite(n) && n >= 0
                    return;
                end
            end
        catch
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
