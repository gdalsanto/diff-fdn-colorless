function [y] = normalizeRMS(x,targetRMS)
    % root mean square normalization
    N = length(x);
    y = sqrt((N*targetRMS^2)./sum(x.^2)).*x;
    
end