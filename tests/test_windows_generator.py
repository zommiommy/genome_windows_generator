from genome_windows_generator import GenomeWindowsGenerator


def test_winodws_generator():
    data_generator = GenomeWindowsGenerator(
        assembly="hg19",
        window_size=200,
        batch_size=3,
        buffer_size=5,
        train_chromosomes=["chr1"],
        test_chromosomes=["chr2"],
        clear_cache=True,
        cache_dir="/tmp",
    )

    next(data_generator.train())

    data_generator.close()

    data_generator = GenomeWindowsGenerator(
        assembly="hg19",
        window_size=200,
        batch_size=3,
        train_chromosomes=["chr1"],
        test_chromosomes=["chr2"],
        cache_dir="/tmp",
    )

    next(data_generator.test())

    data_generator.close()
