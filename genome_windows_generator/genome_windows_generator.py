

import os
import shutil
import itertools
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from dict_hash import sha256
from ucsc_genomes_downloader import Genome
from multiprocessing import Pool, cpu_count

from .one_hot import one_hot_encoder
from .decorators import cache_method
from .tasselize import tasselize_window

# Dataset generator Batchsize_scheduler
#       \                 /
#         \             /
#            \        /
#           Buffer Generator
#                  |
#           _buffer_encoder_generator
#                  |
#           _generator
#


def _model_gaps(gap_mask):
    return np.mean(gap_mask, axis=0), np.cov(gap_mask.T)


def _dataset_generator(dataset):
    while True:
        np.random.shuffle(dataset)
        for value in dataset:
            yield value


class GenomeWindowsGenerator:

    n_types = ["uniform", "normal"]

    def __init__(self,
                 assembly,
                 window_size,
                 batch_size,
                 buffer_size=None,
                 max_gap_size=100,
                 train_chromosomes=None,
                 val_chromosomes=None,
                 cache_dir=None,
                 lazy_load=True,
                 clear_cache=False,
                 compile_on_start=True,
                 n_type="uniform"
                 ):
        self.assembly, self.window_size = assembly, window_size
        self.max_gap_size, self.batch_size, self.val_chromosomes = max_gap_size, batch_size, val_chromosomes

        # Buffersize default None == cpu count for optimal performance:
        if not buffer_size:
            buffer_size = cpu_count()
        self.buffer_size = buffer_size

        # Validate the type of N
        if n_type not in self.n_types:
            raise ValueError("n_type must be one of %s" % n_type)
        self.n_type = n_type

        # Get the cache dir
        cache_dir = cache_dir or os.environ.get("CACHE_PATH", None) or "/tmp"

        self._cache_directory = "/".join([cache_dir,
                                          assembly, str(window_size)])

        if clear_cache:
            self.clean_cache()

        # Generate a pool of processes to save the overhead
        self.workers = max(2, cpu_count())
        self.pool = Pool(self.workers)

        # Preprocess all the possible data
        self.genome = Genome(
            assembly=assembly,
            lazy_load=lazy_load,
            cache_directory=cache_dir,
        )

        if not val_chromosomes:
            self.val_chromosomes = []

        # If no chromosomes passed then use all the genome
        if not train_chromosomes:
            self.chromosomes = sorted(list(self.genome))
        else:
            self.chromosomes = train_chromosomes + self.val_chromosomes

        self.instance_hash = sha256(
            {
                "assembly":self.assembly,
                "chromosomes":self.chromosomes,
                "window_size":self.window_size,
                "max_gap_size":self.max_gap_size,
                "n_type":n_type,
            }
        )

        if compile_on_start:
            self.compile()

    def compile(self):
        filled = self._filled()
        windows = self._tasselize_windows(filled, self.window_size)
        sequences = self._encode_sequences(windows)

        self._windows_train, self._windows_val = self._train_val_split(
            sequences)

        gap_mask = self._render_gaps()
        self._mean, self._cov = _model_gaps(gap_mask)

    def _train_val_split(self, sequences):
        # Get the set of chromosomes
        # TODO do we need a seed here?
        # Find the splitting index
        windows_train = sum(
            [
                sequences[chrom].sequence.tolist()
                for chrom in tqdm(
                    self.chromosomes,
                    desc="Groupping Train windows",
                    leave=False
                )
                if chrom not in self.val_chromosomes
            ], []
        )
        windows_val = sum(
            (
                sequences[chrom].sequence.tolist()
                for chrom in tqdm(
                    self.chromosomes,
                    desc="Groupping val windows",
                    leave=False
                )
                if chrom in self.val_chromosomes
            ), []
        )
        return windows_train, windows_val

    def steps_per_epoch(self):
        return len(self._windows_train) // self.batch_size

    def validation_steps(self):
        return len(self._windows_val) // self.batch_size

    @cache_method("{_cache_directory}/{instance_hash}_filled.pkl")
    def _filled(self):
        return self.genome.filled(chromosomes=self.chromosomes)

    @cache_method("{_cache_directory}/{instance_hash}_gap_mask.pkl")
    def _render_gaps(self):
        # Compute
        gaps = self.genome.gaps(chromosomes=self.chromosomes)
        # Keeping only small gaps
        gaps = gaps[gaps.chromEnd - gaps.chromStart <= self.max_gap_size]
        # Expand windows
        mid_point = ((gaps.chromEnd + gaps.chromStart)/2).astype(int)
        gaps.chromStart = (mid_point - self.window_size/2).astype(int)
        gaps.chromEnd = (mid_point + self.window_size/2).astype(int)
        # Rendering gap sequences
        gapped_sequences = self.genome.bed_to_sequence(gaps)
        # Rendering gap mask
        return np.array([
            np.array(list(sequence.lower())) == "n"
            for sequence in gapped_sequences.sequence
        ])

    @cache_method("{_cache_directory}/{instance_hash}_tasselized.pkl")
    def _tasselize_windows(self, bed: pd.DataFrame, window_size: int):
        # Compute
        tasks = (
            (row.chrom, row.chromStart, row.chromEnd, window_size)
            for _, row in bed.iterrows()
        )
        return pd.concat(list(tqdm(
            self.pool.imap(tasselize_window, tasks),
            total=bed.shape[0],
            desc="Tasselizing windows",
            leave=False
        )))

    @cache_method("{_cache_directory}/{instance_hash}_sequences.pkl")
    def _encode_sequences(self, windows):
        bed = self.genome.bed_to_sequence(windows)
        return {
            chrom: data
            for chrom, data in bed.groupby("chrom")
        }

    def batchsize_scheduler(self):
        while True:
            yield self.batch_size

    def _buffer_generator(self, dataset):
        iterable = _dataset_generator(dataset)
        for batch_size in self.batchsize_scheduler():
            yield [
                list(itertools.islice(iterable, batch_size))
                for _ in range(self.buffer_size)
            ]

    def _buffer_encoder_generator(self, dataset):
        for buffer in self._buffer_generator(dataset):
            yield list(self.pool.imap(one_hot_encoder, buffer))

    def _generator(self, dataset):
        for buffer in self._buffer_encoder_generator(dataset):
            for batch in buffer:
                yield batch

    def generator(self):
        return self._generator(self._windows_train)

    def validation_data(self):
        if not self.val_chromosomes:
            raise ValueError(
                "Can't return the val generator since "
                "no val chromosomes were specified"
            )
        return self._generator(self._windows_val)

    def clean_cache(self):
        if os.path.exists(self._cache_directory):
            shutil.rmtree(self._cache_directory)

    def close(self):
        if "pool" in vars(self):
            self.pool.close()
            self.pool.join()
