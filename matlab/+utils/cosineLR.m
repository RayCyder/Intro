function lr = cosineLR(epoch, numEpochs, baseLR, minLR)
t = (epoch-1) / max(numEpochs-1,1);
lr = minLR + 0.5*(baseLR - minLR)*(1 + cos(pi*t));
end
