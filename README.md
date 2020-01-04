# windows_generator

```python
from windows_generator import WindowsGenerator, NoisyWindowsGenerator

data_generator = NoisyWindowsGenerator(
    assembly="hg19",
    window_size=200,
    batch_size=3,
    buffer_size=5,
    test_chromosomes=["chr1", "chr5"]
)
```

The methods `train, test` returns two independant generator of the train data and test data respectivly.

This is package is mainly meant to be used with `keras`'s `fit_generator`.

```python
model.fit_generator(
    epochs=100,
    steps_per_epoch=len(data_generator),
    generator=data_generator.train(),
    validation_steps=data_generator.test(),
)
```
