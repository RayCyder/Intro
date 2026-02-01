function [dsTrain, dsTest, classNames] = makeCIFAR10Datastores(rootDir)
% Download CIFAR-10 (matlab version) to current folder if missing
url = "https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz";
out = fullfile(pwd, "cifar-10-matlab.tar.gz");
if ~isfile(out)
    websave(out, url);
end
untar(out, pwd);   % extracts cifar-10-batches-mat

trainDir = fullfile(rootDir, "train");
testDir  = fullfile(rootDir, "test");

if ~isfolder(trainDir) || ~isfolder(testDir)
    error("Please prepare CIFAR-10 folders:\n  %s\n  %s\nEach must contain subfolders per class.", trainDir, testDir);
end

dsTrain = imageDatastore(trainDir, "IncludeSubfolders",true, "LabelSource","foldernames");
dsTest  = imageDatastore(testDir,  "IncludeSubfolders",true, "LabelSource","foldernames");

classNames = categories(dsTrain.Labels);
end
