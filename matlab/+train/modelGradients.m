function [loss, grads, logits] = modelGradients(net, X, T)
%modelGradients  Forward + loss + gradients.
%
% Outputs:
%   loss   : scalar dlarray
%   grads  : Learnables table of gradients
%   logits : class scores in 'CB' layout (KxB)
%
% Note: callers can still request only [loss,grads]; logits is optional.

logits = forward(net, X);

% Ensure logits is KxB (CB). Some networks may produce 1x1xKxB.
if ndims(logits) == 4
    logits = squeeze(logits);
end

loss = losses.crossEntropyFromLogits(logits, T);

% Gradients w.r.t. learnable parameters
grads = dlgradient(loss, net.Learnables);
end
