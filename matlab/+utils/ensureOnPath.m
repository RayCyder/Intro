function ensureOnPath()
rootDir = fileparts(fileparts(mfilename('fullpath')));
if ~contains(path, rootDir)
    addpath(rootDir);
end
end
