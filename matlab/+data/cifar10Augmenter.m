
function augmentFcn = cifar10Augmenter(varargin)
% cifar10Augmenter  CIFAR-10 training augmentation (PyTorch-aligned).
%
% Returns a function handle that applies:
%   1) RandomHorizontalFlip (p=0.5)
%   2) RandomCrop(32, padding=4) with zero padding
%
% Usage:
%   aug = data.cifar10Augmenter();
%   Xaug = aug(X);   % X: 32x32x3xB
%
% Name-value options:
%   'OutputSize' : [H W] output crop size (default [32 32])
%   'Padding'    : scalar padding size (default 4)
%   'FlipProb'   : probability of horizontal flip (default 0.5)

p = inputParser;
addParameter(p, 'OutputSize', [32 32]);
addParameter(p, 'Padding', 4);
addParameter(p, 'FlipProb', 0.5);
parse(p, varargin{:});

outSize  = p.Results.OutputSize;
padding  = p.Results.Padding;
flipProb = p.Results.FlipProb;

augmentFcn = @(X) augmentBatchCIFAR10(X, outSize, padding, flipProb);
end

% ===================== helper =====================

function X = augmentBatchCIFAR10(X, outSize, padding, flipProb)
% X: HxWxCxB numeric array (uint8/single/double)

if iscell(X)
    X = cat(4, X{:});
end

H = outSize(1);
W = outSize(2);

% Basic checks
if ndims(X) ~= 4
    error('augmentBatchCIFAR10 expects a 4-D array HxWxCxB.');
end

[Hin, Win, Cin, B] = size(X);
if Hin ~= H || Win ~= W
    % We assume CIFAR input is 32x32. If different, we still crop to outSize.
    % (This keeps behavior predictable.)
end

Xout = zeros(H, W, Cin, B, 'like', X);

for i = 1:B
    img = X(:,:,:,i);

    % 1) RandomHorizontalFlip
    if rand() < flipProb
        img = fliplr(img);
    end

    % 2) RandomCrop with zero padding
    if padding > 0
        % Zero padding matches torchvision RandomCrop(padding=4) default.
        imgPad = padarray(img, [padding padding], 0, 'both');
        % Choose top-left corner uniformly
        y0 = randi([1, 2*padding + 1]);
        x0 = randi([1, 2*padding + 1]);
        img = imgPad(y0:y0+H-1, x0:x0+W-1, :);
    else
        % If no padding, just ensure the crop is correct size
        img = img(1:H, 1:W, :);
    end

    Xout(:,:,:,i) = img;
end

X = Xout;
end
