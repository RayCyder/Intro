function [X, T] = preprocessMiniBatch(Xcell, Y, classNames, cfg)
% Xcell is a cell array of images
X = cat(4, Xcell{:});              % HxWxCxB
X = im2single(X);                  % float32

% PyTorch CIFAR normalize defaults
meanRGB = reshape(single([0.4914 0.4822 0.4465]), 1,1,3);
stdRGB  = reshape(single([0.2023 0.1994 0.2010]),  1,1,3);
X = (X - meanRGB) ./ stdRGB;

X = dlarray(X, "SSCB");

% labels -> onehot (C,B)
Y = categorical(Y, classNames);
T = onehotencode(Y, 1, "ClassNames", classNames);  % CxB
T = dlarray(single(T), "CB");

% cast
X = cast(X, cfg.precision);
T = cast(T, cfg.precision);
end
