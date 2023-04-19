samplerate = 48000
dir_path = "./output"

# dataset
num = 256                   # dataaset size
min_nfft = 8*samplerate     # min number of fft points M_min
max_nfft = 10*samplerate    # max number of fft points M_max
split = 0.78125             # trianing / validation split
shuffle = True              # if on shuffle the data after every epoch


# training
batch_size = 4
max_epochs = 15


# optimizer
lr = 1e-3                   # leaning rate
alpha = 300                 # temporal loss scaling factor

# netowrk 
gain_per_sample = 0.9999    # gamma
delays = [809., 877., 937., 1049., 1151., 1249., 1373., 1499.]
learnDelays = delays    # one between [delays, 'init']
N = len(delays)

# [1499., 1889., 2381., 2999.]
# [797., 839., 2381., 2999.]

# [997., 1153., 1327., 1559., 1801., 2099.]
# [887., 911., 941., 1699., 1951., 2053.]

# [809., 877., 937., 1049., 1151., 1249., 1373., 1499.]
# [241., 263., 281., 293., 1193., 1319., 1453., 1597.]