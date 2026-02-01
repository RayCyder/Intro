function [loss, grads] = modelGradients(net, X, T)
logits = forward(net, X);
if ndims(logits)==4
    logits = squeeze(logits);
end
loss = losses.crossEntropyFromLogits(logits, T);
grads = dlgradient(loss, net.Learnables);
end
