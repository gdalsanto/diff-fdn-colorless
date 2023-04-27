function loss = asyPLoss(yPred, yTrue)
    % asymetric mean squared* error between yPred and yTrue
    exp = 2*ones(size(yPred)) + 2*(abs(yPred) > abs(yTrue)); % exponent
    loss = mean((abs(yPred)-abs(yTrue)).^exp);
end