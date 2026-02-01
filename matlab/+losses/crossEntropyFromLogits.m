function loss = crossEntropyFromLogits(logits, T)
%crossEntropyFromLogits  Cross-entropy loss from unnormalized logits.
%
% This helper supports both:
%   (1) formatted dlarray logits (e.g., 'CB')  -> call softmax(logits) WITHOUT DataFormat
%   (2) unformatted numeric logits            -> call softmax(logits,'DataFormat','CB')
%
% Expected shapes:
%   logits: KxB (or 1x1xKxB)
%   T:      KxB one-hot (or dlarray with format 'CB')

% Squeeze common classifier head output: 1x1xKxB -> KxB
if ndims(logits) == 4
    logits = squeeze(logits);
end

% Detect whether logits is a formatted dlarray
isFormatted = false;
if isa(logits, 'dlarray')
    try
        fmt = dims(logits);
        isFormatted = ~isempty(fmt);
    catch
        isFormatted = false;
    end
end

% Softmax
if isFormatted
    % When logits is formatted dlarray, DataFormat option is NOT supported.
    P = softmax(logits);
else
    % For unformatted inputs, specify the intended layout.
    P = softmax(logits, 'DataFormat', 'CB');
end

% Cross-entropy with numerical stability
loss = -mean(sum(T .* log(P + 1e-12), 1));
end
