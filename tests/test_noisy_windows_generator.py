from windows_generator import NoisyWindowsGenerator
data_generator = NoisyWindowsGenerator(
    assembly="sacCer3",
    window_size=200,
    batch_size=3,
    buffer_size=5
)

x, y = next(data_generator.train())