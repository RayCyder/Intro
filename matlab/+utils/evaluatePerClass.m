function [accOverall, avgLoss, accPerClass, countPerClass] = evaluatePerClass(net, mbq, useGPU, classNames)
%evaluatePerClass  Evaluate classification accuracy/loss overall and per class.
%
% Inputs:
%   net        : dlnetwork
%   mbq        : minibatchqueue yielding (X,T) with formats {'SSCB','CB'}
%   useGPU     : logical
%   classNames : (optional) 1xK cellstr for printing
%
% Outputs:
%   accOverall    : overall accuracy in [0,1]
%   avgLoss       : average cross-entropy loss
%   accPerClass   : Kx1 per-class accuracy in [0,1] (NaN if class absent)
%   countPerClass : Kx1 per-class sample counts

reset(mbq);

numCorrect = 0;
numTotal = 0;
lossSum = 0;
numBatches = 0;

K = [];
correctPerClass = [];
countPerClass = [];

while hasdata(mbq)
    [X, T] = next(mbq);
    if useGPU
        X = gpuArray(X);
        T = gpuArray(T);
    end

    logits = forward(net, X);
    if ndims(logits) == 4
        logits = squeeze(logits);
    end

    % Softmax: if logits is a formatted dlarray, do NOT pass 'DataFormat'.
    isFormatted = false;
    if isa(logits,'dlarray')
        try
            isFormatted = ~isempty(dims(logits));
        catch
            isFormatted = false;
        end
    end

    if isFormatted
        P = softmax(logits);
    else
        P = softmax(logits, 'DataFormat', 'CB');
    end

    loss = -mean(sum(T .* log(P + 1e-12), 1));

    [~, pred] = max(gather(extractdata(P)), [], 1); % 1xB
    [~, gt]   = max(gather(extractdata(T)), [], 1); % 1xB

    if isempty(K)
        K = size(gather(extractdata(T)), 1);
        correctPerClass = zeros(K,1);
        countPerClass   = zeros(K,1);
    end

    numCorrect = numCorrect + sum(pred == gt);
    numTotal   = numTotal + numel(gt);

    for c = 1:K
        mask = (gt == c);
        if any(mask)
            countPerClass(c)   = countPerClass(c) + sum(mask);
            correctPerClass(c) = correctPerClass(c) + sum(pred(mask) == c);
        end
    end

    lossSum = lossSum + double(gather(extractdata(loss)));
    numBatches = numBatches + 1;
end

accOverall = numCorrect / max(numTotal,1);
avgLoss = lossSum / max(numBatches,1);

accPerClass = correctPerClass ./ max(countPerClass, 1);
accPerClass(countPerClass == 0) = NaN;

% Optional printing (Python-like)
if nargin >= 4 && ~isempty(classNames)
    fprintf('Per-class accuracy:\n');
    for c = 1:numel(classNames)
        if c <= numel(accPerClass)
            if isnan(accPerClass(c))
                fprintf('  %s: (no samples)\n', classNames{c});
            else
                fprintf('  %s: %.2f%% (%d)\n', classNames{c}, accPerClass(c)*100, countPerClass(c));
            end
        end
    end
    fprintf('Overall Acc: %.2f%% | Avg Loss: %.4f\n', accOverall*100, avgLoss);
end

end