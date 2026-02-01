
function [dsTrain, dsVal, dsTest, classNames] = makeCIFAR10Datastores(rootDir, varargin)
% makeCIFAR10Datastores  Prepare CIFAR-10 datastores (arrayDatastore) without writing PNGs.
%
% This function mirrors the common PyTorch pipeline more closely:
%   - Downloads CIFAR-10 (matlab version) if missing
%   - Loads the 50k training + 10k test images into memory
%   - Splits training into train/val by 9:1 (default)
%   - Returns combined datastores: (images, labels)
%
% Outputs:
%   dsTrain: combined datastore yielding (X, Y) for training
%   dsVal:   combined datastore yielding (X, Y) for validation
%   dsTest:  combined datastore yielding (X, Y) for testing
%   classNames: cellstr of the 10 class names (fixed order)
%
% Optional name-value pairs:
%   "ValRatio" : fraction for validation split (default 0.1)
%   "Seed"     : RNG seed for split reproducibility (default 888)
%
% Notes:
%   - Images are returned as uint8 arrays of size 32x32x3xB.
%   - Labels are returned as categorical with categories = classNames.

p = inputParser;
addParameter(p, 'ValRatio', 0.1);
addParameter(p, 'Seed', 888);
parse(p, varargin{:});
valRatio = p.Results.ValRatio;
seed = p.Results.Seed;

% Download CIFAR-10 (matlab version) to rootDir if missing
if ~isfolder(rootDir)
    mkdir(rootDir);
end

url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
out = fullfile(rootDir, 'cifar-10-matlab.tar.gz');
if ~isfile(out)
    websave(out, url);
end

cifarMatDir = fullfile(rootDir, 'cifar-10-batches-mat');
if ~isfolder(cifarMatDir)
    untar(out, rootDir);   % extracts cifar-10-batches-mat
end

% Read class names
meta = load(fullfile(cifarMatDir, 'batches.meta.mat'), 'label_names');
classNames = cellstr(meta.label_names);

% Load all training batches into arrays
[trainX, trainYNum] = loadCifarSplit(cifarMatDir, true);
[testX,  testYNum]  = loadCifarSplit(cifarMatDir, false);

% Convert numeric labels (0..9) to categorical with fixed category order
trainY = categorical(classNames(double(trainYNum)+1), classNames);
testY  = categorical(classNames(double(testYNum)+1),  classNames);

% Split train into train/val by valRatio with per-class stratification (PyTorch-aligned)
N = size(trainX, 4);
valRatio = max(0, min(0.5, valRatio));

rng(seed);
trainIdx = [];
valIdx   = [];

% Stratify using numeric labels 0..9 (stored in trainYNum)
for k = 0:9
    idxK = find(trainYNum == k);
    nk = numel(idxK);
    if nk == 0
        continue;
    end

    %permK = idxK(randperm(nk));
    permK = idxK; % keep class order (no shuffle)
    numValK = round(nk * valRatio);

    % Ensure both splits are non-empty when possible
    if nk >= 2
        numValK = min(max(numValK, 1), nk-1);
    else
        numValK = min(numValK, nk);
    end

    valIdx   = [valIdx;   permK(1:numValK)];
    trainIdx = [trainIdx; permK(numValK+1:end)];
end

% Shuffle final indices so batches mix classes
trainIdx = trainIdx(randperm(numel(trainIdx)));
valIdx   = valIdx(randperm(numel(valIdx)));

trainX_ = trainX(:,:,:,trainIdx);
trainY_ = trainY(trainIdx);

valX_ = trainX(:,:,:,valIdx);
valY_ = trainY(valIdx);

% Build arrayDatastore pipelines
% X datastore iterates along 4th dim (batch dimension)
% Y datastore iterates along rows

dsTrain = combine( ...
    arrayDatastore(trainX_, 'IterationDimension', 4), ...
    arrayDatastore(trainY_) ...
);

dsVal = combine( ...
    arrayDatastore(valX_, 'IterationDimension', 4), ...
    arrayDatastore(valY_) ...
);

dsTest = combine( ...
    arrayDatastore(testX, 'IterationDimension', 4), ...
    arrayDatastore(testY) ...
);

end

% ===================== helpers =====================

function [X, Y] = loadCifarSplit(cifarMatDir, isTrain)
% Return X: 32x32x3xN uint8, Y: Nx1 numeric labels (0..9)
if isTrain
    N = 50000;
    X = zeros(32,32,3,N,'uint8');
    Y = zeros(N,1,'int32');
    idx = 1;
    for b = 1:5
        batchPath = fullfile(cifarMatDir, sprintf('data_batch_%d.mat', b));
        S = load(batchPath, 'data', 'labels');
        imgs = reshapeCifarBatch(S.data);
        n = size(imgs,4);
        X(:,:,:,idx:idx+n-1) = imgs;
        Y(idx:idx+n-1) = int32(S.labels);
        idx = idx + n;
    end
else
    batchPath = fullfile(cifarMatDir, 'test_batch.mat');
    S = load(batchPath, 'data', 'labels');
    X = reshapeCifarBatch(S.data);
    Y = int32(S.labels);
end
end

function imgs = reshapeCifarBatch(data)
% data: N x 3072 uint8
% CIFAR ordering: [R(1024), G(1024), B(1024)] per row, row-major for 32x32.
% Vectorized reshape: transpose -> reshape -> permute to correct orientation.
N = size(data,1);
X = reshape(data', 32,32,3,N);   % 32x32x3xN (column-major)
imgs = permute(X, [2 1 3 4]);    % swap H/W to match standard image orientation
end
