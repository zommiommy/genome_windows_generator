from windows_generator import NoisyWindowsGenerator


def test_noisy_windows_generator():
    data_generator = NoisyWindowsGenerator(
        assembly="hg19",
        window_size=200,
        batch_size=3,
        buffer_size=5,
        train_chromosomes=["chr1"],
        test_chromosomes=["chr2"],
        cache_dir="/tmp",
        n_type="uniform"
    )

    next(data_generator.train())
    next(data_generator.test())

    print(len(data_generator))

    data_generator.close()

    data_generator = NoisyWindowsGenerator(
        assembly="hg19",
        window_size=200,
        batch_size=3,
        buffer_size=5,
        train_chromosomes=["chr1"],
        test_chromosomes=["chr2"],
        cache_dir="/tmp",
        n_type="normal"
    )

    next(data_generator.train())
    next(data_generator.test())

    print(len(data_generator))

    data_generator.close()
