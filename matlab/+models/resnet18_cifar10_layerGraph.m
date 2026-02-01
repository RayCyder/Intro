function lgraph = resnet18_cifar10_layerGraph(numClasses, inputSize)
%RESNET18_CIFAR10_LAYERGRAPH  ResNet-18 (CIFAR10 style) aligned to 131.py.
%
% Python (131.py) spec:
% - conv1: 3x3, 64, stride=1, pad=1, bias=False
% - no maxpool
% - BasicBlock, layers [2,2,2,2]
% - downsample in layer2/3/4 first block with stride=2, projection 1x1 conv + BN
% - avg_pool2d(kernel=4) then flatten then linear
% - output logits (no softmax)
%
% Default:
%   numClasses = 10
%   inputSize  = [32 32 3]
%
% Note: MATLAB conv layers always have a Bias parameter; we freeze it at 0
% to be functionally equivalent to bias=False in PyTorch.

if nargin < 1 || isempty(numClasses), numClasses = 10; end
if nargin < 2 || isempty(inputSize),  inputSize  = [32 32 3]; end

bnEps   = 1e-5;
bnDecay = 0.1; % match PyTorch BN momentum=0.1 behavior  [oai_citation:2‡MathWorks](https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.batchnormalizationlayer.html)

% -------- Stem (3x3 stride=1, no maxpool) --------
layersStem = [
    imageInputLayer(inputSize, "Name","input", "Normalization","none")

    convolution2dLayer(3, 64, ...
        "Stride", 1, "Padding","same", ...
        "BiasInitializer","zeros", ...
        "BiasLearnRateFactor",0, "BiasL2Factor",0, ...
        "Name","stem_conv")

    batchNormalizationLayer("Name","stem_bn", ...
        "Epsilon",bnEps, "MeanDecay",bnDecay, "VarianceDecay",bnDecay)

    reluLayer("Name","stem_relu")
];

lgraph = layerGraph(layersStem);

inName = "stem_relu";

% -------- Stages: [2,2,2,2] --------
% stage1: 64, no downsample
[lgraph, inName] = addBasicBlock(lgraph, "s1_b1", inName, 64, 1, false, bnEps, bnDecay);
[lgraph, inName] = addBasicBlock(lgraph, "s1_b2", inName, 64, 1, false, bnEps, bnDecay);

% stage2: 128, downsample at first block
[lgraph, inName] = addBasicBlock(lgraph, "s2_b1", inName, 128, 2, true,  bnEps, bnDecay);
[lgraph, inName] = addBasicBlock(lgraph, "s2_b2", inName, 128, 1, false, bnEps, bnDecay);

% stage3: 256
[lgraph, inName] = addBasicBlock(lgraph, "s3_b1", inName, 256, 2, true,  bnEps, bnDecay);
[lgraph, inName] = addBasicBlock(lgraph, "s3_b2", inName, 256, 1, false, bnEps, bnDecay);

% stage4: 512
[lgraph, inName] = addBasicBlock(lgraph, "s4_b1", inName, 512, 2, true,  bnEps, bnDecay);
[lgraph, inName] = addBasicBlock(lgraph, "s4_b2", inName, 512, 1, false, bnEps, bnDecay);

% -------- Head: avg_pool2d(kernel=4) -> flatten -> linear (logits) --------
layersHead = [
    averagePooling2dLayer(4, "Stride",4, "Name","avgpool")  % match F.avg_pool2d(out,4)
    flattenLayer("Name","flatten")
    fullyConnectedLayer(numClasses, "Name","linear")        % match nn.Linear
];

lgraph = addLayers(lgraph, layersHead);
lgraph = connectLayers(lgraph, inName, "avgpool");

end


% ========================================================================
% Helper: CIFAR BasicBlock aligned to 131.py
% ========================================================================
function [lgraph, outName] = addBasicBlock(lgraph, blockName, inName, numFilters, stride, useProjection, bnEps, bnDecay)
% main: conv3x3(stride) -> BN -> ReLU -> conv3x3(1) -> BN
% skip: identity OR conv1x1(stride) -> BN
% out: add -> ReLU

mainLayers = [
    convolution2dLayer(3, numFilters, ...
        "Stride", stride, "Padding","same", ...
        "BiasInitializer","zeros", ...
        "BiasLearnRateFactor",0, "BiasL2Factor",0, ...
        "Name", blockName + "_conv1")

    batchNormalizationLayer("Name", blockName + "_bn1", ...
        "Epsilon",bnEps, "MeanDecay",bnDecay, "VarianceDecay",bnDecay)

    reluLayer("Name", blockName + "_relu1")

    convolution2dLayer(3, numFilters, ...
        "Stride", 1, "Padding","same", ...
        "BiasInitializer","zeros", ...
        "BiasLearnRateFactor",0, "BiasL2Factor",0, ...
        "Name", blockName + "_conv2")

    batchNormalizationLayer("Name", blockName + "_bn2", ...
        "Epsilon",bnEps, "MeanDecay",bnDecay, "VarianceDecay",bnDecay)
];

addLayer = additionLayer(2, "Name", blockName + "_add");
outRelu  = reluLayer("Name", blockName + "_out");

lgraph = addLayers(lgraph, mainLayers);
lgraph = addLayers(lgraph, addLayer);
lgraph = addLayers(lgraph, outRelu);

% connect input -> main
lgraph = connectLayers(lgraph, inName, blockName + "_conv1");
% main -> add/in1
lgraph = connectLayers(lgraph, blockName + "_bn2", blockName + "_add/in1");

% shortcut
if useProjection
    projLayers = [
        convolution2dLayer(1, numFilters, ...
            "Stride", stride, "Padding","same", ...
            "BiasInitializer","zeros", ...
            "BiasLearnRateFactor",0, "BiasL2Factor",0, ...
            "Name", blockName + "_proj_conv")
        batchNormalizationLayer("Name", blockName + "_proj_bn", ...
            "Epsilon",bnEps, "MeanDecay",bnDecay, "VarianceDecay",bnDecay)
    ];
    lgraph = addLayers(lgraph, projLayers);
    lgraph = connectLayers(lgraph, inName, blockName + "_proj_conv");
    lgraph = connectLayers(lgraph, blockName + "_proj_bn", blockName + "_add/in2");
else
    lgraph = connectLayers(lgraph, inName, blockName + "_add/in2");
end

% add -> relu
lgraph = connectLayers(lgraph, blockName + "_add", blockName + "_out");

outName = blockName + "_out";
end