genome_windows_generator
=========================================================================================
|travis| |sonar_quality| |sonar_maintainability| |codacy| |code_climate_maintainability| |pip| |downloads|

How do I install this package?
----------------------------------------------
As usual, just download it using pip:

.. code:: shell

    pip install genome_windows_generator

Tests Coverage
----------------------------------------------
Since some software handling coverages sometime get slightly different results, here's three of them:

|coveralls| |sonar_coverage| |code_climate_coverage|


.. code:: python

    from genome_windows_generator import GenomeWindowsGenerator, NoisyWindowsGenerator

    dg = NoisyWindowsGenerator(
        assembly="hg19",
        window_size=200,
        batch_size=3,
        buffer_size=5,
        test_chromosomes=["chr1", "chr5"]
    )





The methods `train, test` returns two independant generator of the train data and test data respectivly.



This is package is mainly meant to be used with `keras`'s `fit_generator`.


.. code:: python

    model.fit_generator(
        epochs=100,
        generator=dg.generator(),
        steps_per_epoch=dg.steps_per_epoch(),
        validation_data=dg.validation_data(),
        validation_steps=dg.validation_steps(),
    )



.. |travis| image:: https://travis-ci.org/zommiommy/genome_windows_generator.png
   :target: https://travis-ci.org/zommiommy/genome_windows_generator
   :alt: Travis CI build

.. |sonar_quality| image:: https://sonarcloud.io/api/project_badges/measure?project=zommiommy_genome_windows_generator&metric=alert_status
    :target: https://sonarcloud.io/dashboard/index/zommiommy_genome_windows_generator
    :alt: SonarCloud Quality

.. |sonar_maintainability| image:: https://sonarcloud.io/api/project_badges/measure?project=zommiommy_genome_windows_generator&metric=sqale_rating
    :target: https://sonarcloud.io/dashboard/index/zommiommy_genome_windows_generator
    :alt: SonarCloud Maintainability

.. |sonar_coverage| image:: https://sonarcloud.io/api/project_badges/measure?project=zommiommy_genome_windows_generator&metric=coverage
    :target: https://sonarcloud.io/dashboard/index/zommiommy_genome_windows_generator
    :alt: SonarCloud Coverage

.. |coveralls| image:: https://coveralls.io/repos/github/zommiommy/genome_windows_generator/badge.svg
    :target: https://coveralls.io/github/zommiommy/genome_windows_generator
    :alt: Coveralls Coverage

.. |pip| image:: https://badge.fury.io/py/genome-windows-generator.svg
    :target: https://badge.fury.io/py/genome-windows-generator
    :alt: Pypi project

.. |downloads| image:: https://pepy.tech/badge/genome-windows-generator
    :target: https://pepy.tech/badge/genome-windows-generator
    :alt: Pypi total project downloads 

.. |codacy|  image:: https://api.codacy.com/project/badge/Grade/8dd7ef7604084ded82ae70acddc16264
    :target: https://www.codacy.com/manual/zommiommy/genome_windows_generator?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=zommiommy/genome_windows_generator&amp;utm_campaign=Badge_Grade
    :alt: Codacy Maintainability

.. |code_climate_maintainability| image:: https://api.codeclimate.com/v1/badges/4e850c49fac5b73cab29/maintainability
    :target: https://codeclimate.com/github/zommiommy/genome_windows_generator/maintainability
    :alt: Maintainability

.. |code_climate_coverage| image:: https://api.codeclimate.com/v1/badges/4e850c49fac5b73cab29/test_coverage
    :target: https://codeclimate.com/github/zommiommy/genome_windows_generator/test_coverage
    :alt: Code Climate Coverate
