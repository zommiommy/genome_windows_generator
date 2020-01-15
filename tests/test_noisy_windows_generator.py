from genome_windows_generator import NoisyWindowsGenerator


def test_noisy_genome_windows_generator():
    data_generator = NoisyWindowsGenerator(
        assembly="hg19",
        window_size=200,
        batch_size=3,
        buffer_size=5,
        train_chromosomes=["chr1"],
        val_chromosomes=["chr2"],
        cache_dir="/tmp",
        n_type="uniform"
    )

    next(data_generator.generator())
    next(data_generator.validation_data())

    print(data_generator.steps_per_epoch())
    print(data_generator.validation_steps())

    data_generator.close()

    data_generator = NoisyWindowsGenerator(
        assembly="hg19",
        window_size=200,
        batch_size=3,
        buffer_size=5,
        train_chromosomes=["chr1"],
        val_chromosomes=["chr2"],
        cache_dir="/tmp",
        n_type="normal"
    )

    data_generator.close()
