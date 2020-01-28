function [bestScore, bestThreshold] = computeBestThreshold(DT,values, labels)
    % candidate threshold sampled from this values gaussian
    thresholds = randn(DT.numThreshold, 1).*std(values) + mean(values);
    N = numel(values);
    infogains = ones(DT.numThreshold, 1)*-Inf;

    for i = 1:DT.numThreshold
        toLeft = values < thresholds(i);
        numL = numel(labels(toLeft)) + DT.DIRICHLET;
        numR = numel(labels(~toLeft)) + DT.DIRICHLET;
        LDist = 1/numL.*(hist(labels(toLeft), 1:DT.numClass) ...
                         + DT.DIRICHLET./DT.numClass);
        RDist = 1/numR.*(hist(labels(~toLeft), 1:DT.numClass) ...
                         + DT.DIRICHLET./DT.numClass);
        % calc entropy
        EL = -sum(LDist.*log2(LDist));
        ER = -sum(RDist.*log2(RDist));
        infogains(i) = -1/N*(numL*EL + numR*ER);
    end
    [bestScore, bestind] = max(infogains);
    bestThreshold = thresholds(bestind);
end