function opt = updateLastGrad(opt, gradsOld)
%updateLastGrad  Cache gradients from the "old" network for exact MVR2.
%
% This function stores per-parameter gradients in opt.LastGrad (cell array)
% aligned with net.Learnables ordering used in init/step.
%
% Inputs:
%   opt     : optimizer state (from optimizers.muonmvr2.init)
%   gradsOld: gradients table (same layout as net.Learnables), typically
%             computed on netOld when opt.IsApprox == false.
%
% Output:
%   opt     : updated optimizer state

if ~isstruct(opt) || ~isfield(opt,'LastGrad')
    error('muonmvr2:updateLastGrad:InvalidOpt', 'opt must be a struct with field LastGrad.');
end

if ~istable(gradsOld) || ~any(strcmp(gradsOld.Properties.VariableNames,'Value'))
    error('muonmvr2:updateLastGrad:InvalidGrads', 'gradsOld must be a table with a Value column.');
end

n = height(gradsOld);

% Ensure opt.LastGrad is a cell array with correct length.
if ~iscell(opt.LastGrad) || numel(opt.LastGrad) ~= n
    opt.LastGrad = cell(n,1);
end

for i = 1:n
    g = gradsOld.Value{i};
    if isempty(g)
        opt.LastGrad{i} = [];
    else
        % Store as dlarray to match arithmetic in step.m
        if isa(g,'dlarray')
            opt.LastGrad{i} = g;
        else
            opt.LastGrad{i} = dlarray(g);
        end
    end
end

end
