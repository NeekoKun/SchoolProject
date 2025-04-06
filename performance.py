import pstats
stats = pstats.Stats('profile.out')
stats.strip_dirs().sort_stats('cumulative').print_stats(20)