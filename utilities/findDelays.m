function m = findDelays(startVal, endVal, N)
    % Delays that are prime numbers logarithmically distributed between 
    % startVal and endVal (in samples)
    % N: number of delays 
    candidates = zeros(1,N);
    for i = 0:N-1
        candidates(i+1) = startVal*(endVal/startVal)^(i/(N-1));
    end
    m = zeros(1,N);
    % find the closest prime numbers 
    for i = 1:N
        primeNums = primes(candidates(i));
        % take the larger prime number among the list 
        m(i) = primeNums(end);
    end
end
