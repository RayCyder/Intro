function [dsTrain, dsTest, classNames] = makeCIFAR10Datastores(rootDir)
trainDir = fullfile(rootDir, "train");
testDir  = fullfile(rootDir, "test");

if ~isfolder(trainDir) || ~isfolder(testDir)
    error("Please prepare CIFAR-10 folders:\n  %s\n  %s\nEach must contain subfolders per class.", trainDir, testDir);
end

dsTrain = imageDatastore(trainDir, "IncludeSubfolders",true, "LabelSource","foldernames");
dsTest  = imageDatastore(testDir,  "IncludeSubfolders",true, "LabelSource","foldernames");

classNames = categories(dsTrain.Labels);
end
