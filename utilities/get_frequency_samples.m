function freq_samples = get_frequency_samples(num)
    angle = [0:1/num:1-1/num]*pi;
    abs = ones(1,num);
    freq_samples = abs.*cos(angle)+1i*abs.*sin(angle);
end