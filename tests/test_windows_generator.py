from windows_generator import WindowsGenerator
from ucsc_genomes_downloader import Genome

data_generator = WindowsGenerator(
    assembly="sacCer3",
    window_size=1,
    batch_size=3,
    buffer_size=5,
    cache_dir="/tmp",
)

x, y = next(data_generator.train())