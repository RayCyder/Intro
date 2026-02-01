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
