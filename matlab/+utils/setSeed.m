function setSeed(seed)
if nargin < 1 || isempty(seed)
    seed = 0;
end
rng(seed, "twister");
end
