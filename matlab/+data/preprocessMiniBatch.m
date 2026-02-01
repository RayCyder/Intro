function [X, T] = preprocessMiniBatch(X, Y, classNames, cfg, varargin)
%preprocessMiniBatch  Unified CIFAR-10 minibatch preprocessing for MATLAB training loop.
%
% Supports arrayDatastore/minibatchqueue outputs where X can be:
%   - 4-D numeric array HxWxCxB
%   - cell array of images (each HxWxCx1)
% and Y can be:
%   - categorical array (B-by-1)
%   - numeric labels (0..9)
%   - cell array (often  B-by-1 cell) of scalar categoricals / numerics / strings
%
% Name-value options:
%   'Augmenter' : function handle aug(X)->X for TRAIN ONLY (default [])

p = inputParser;
addParameter(p, 'Augmenter', []);
parse(p, varargin{:});
aug = p.Results.Augmenter;

% ---- X: ensure 4-D array ----
if iscell(X)
    X = cat(4, X{:});
end

% ---- TRAIN augmentation (optional) ----
if ~isempty(aug)
    X = aug(X);
end

% ---- Normalize (PyTorch-aligned) ----
X = im2single(X);
meanRGB = reshape(single([0.4914 0.4822 0.4465]), 1,1,3);
stdRGB  = reshape(single([0.2467 0.2432 0.2611]),  1,1,3);
X = (X - meanRGB) ./ stdRGB;

X = dlarray(X, 'SSCB');

% ---- Labels: robust handling ----
if iscell(Y)
    if isempty(Y)
        Y = categorical([], classNames);
    elseif all(cellfun(@iscategorical, Y))
        % Cell array of scalar categoricals -> cellstr names
        Y = categorical(cellfun(@char, Y, 'UniformOutput', false), classNames);
    elseif all(cellfun(@isnumeric, Y))
        Yn = cell2mat(Y);
        Y = categorical(classNames(double(Yn)+1), classNames);
    else
        % Assume cellstr / string-like
        Y = categorical(Y, classNames);
    end
end

if ~iscategorical(Y)
    if isnumeric(Y)
        % CIFAR-10 labels commonly 0..9
        Y = categorical(classNames(double(Y)+1), classNames);
    else
        Y = categorical(Y, classNames);
    end
end

% Align categories/order to classNames
Y = setcats(Y, classNames);
Y = reordercats(Y, classNames);

if any(isundefined(Y))
    error('Found undefined labels in mini-batch. Check classNames alignment.');
end

% ---- One-hot: output KxB for 'CB' format ----
Y = Y(:);  % ensure Bx1 (dim2 singleton)
Th = onehotencode(Y, 2, 'ClassNames', classNames); % BxK
Th = Th.';                                         % KxB
T = dlarray(single(Th), 'CB');

% ---- Cast ----
X = cast(X, cfg.precision);
T = cast(T, cfg.precision);
end
