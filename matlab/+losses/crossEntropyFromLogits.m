function loss = crossEntropyFromLogits(logits, T)
P = softmax(logits, "DataFormat","CB");
loss = -mean(sum(T .* log(P + 1e-12), 1));
end
