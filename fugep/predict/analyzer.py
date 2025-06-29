'''
Created on May 27, 2021

@author: jsun
'''

from abc import ABCMeta
import os

from ..data import Genome
from .ana_hdl import LogitScoreHandler, AbsDiffScoreHandler, DiffScoreHandler
from .ana_hdl import StdHandler, MeanGVEHandler, PvalHandler
from .ana_hdl import WritePredictionsHandler, WriteRefAltHandler
from .utils import _pad_sequence
from .utils import _truncate_sequence
from .utils import createFilePathWithPrefix

from ..utils import initialize_logger
from ..data.utils import formatChrom
from ..model.nn.utils import load_model

class Analyzer(metaclass = ABCMeta):
    '''
    The base class for applying a trained model to make 
    predictions and perform analysis
    
    Parameters
    ----------
    model : PredMWrapper
        A sequence-based model architecture.
    trainedModelPath : str or list(str)
        The path(s) to the weights file for a trained sequence-based model.
        For a single path, the model architecture must match `model`. For
        a list of paths, assumes that the `model` passed in is of type
        `fugep.utils.MultiModelWrapper`, which takes in a list of
        models. The paths must be ordered the same way the models
        are ordered in that list. `list(str)` input is an API-only function--
        Selene's config file CLI does not support the `MultiModelWrapper`
        functionality at this time    features : list(str)
        The names of the features that the model is predicting.
    analysis : list(str)
        A list of the data files to output. Must input 1 or more of the
        following options: ["abs_diffs", "diffs", "logits", "predictions"].
    outputPath : str
        The path to save results. Can be a directory, or a path with prefix 
        for the output files
    colNamesOfIds : list(str)
        Specify the names of columns that will be used to identify the
        sequence for which FuGEP has made predictions (e.g. (chrom,
        pos, id, ref, alt) will be the column names for variant effect
        prediction outputs). 
    outputFormat : {'tsv', 'hdf5'}
            The desired output format. Currently FuGEP supports TSV and HDF5
            formats.   
    outputSize : int, optional
        The total number of rows in the output. Must be specified when
        the outputFormat is hdf5.
    mode : {'prediction', 'ism', 'varianteffect'}
        If saving model predictions, the handler FuGEP chooses for the
        task is dependent on the mode. For example, the reporter for
        variant effect prediction writes paired ref and alt predictions
        to different files.
    batchSize : int, optional
        Default is 64. The size of the mini-batches to use.
    useCuda : bool, optional
        Default is `False`. Specifies whether CUDA-enabled GPUs are available
        for torch to use.
    dataParallel : bool, optional
        Default is `False`. Specify whether multiple GPUs are available for
        torch to use during training.
    refSeq : class, optional
        Default is `fugep.sequences.Genome`. The type of sequence on
        which this analysis will be performed. Please note that if you need
        to use variant effect prediction, you cannot only pass in the
        class--you must pass in the constructed `fugep.sequences.Sequence`
        object with a particular sequence version (e.g. `Genome("hg19.fa")`).
        This version does NOT have to be the same sequence version that the
        model was trained on. That is, if the sequences in your variants file
        are hg19 but your model was trained on hg38 sequences, you should pass
        in hg19.
    writeMemLimit : int, optional
        Default is 5000. Specify, in MB, the amount of memory you want to
        allocate to storing model predictions/scores. When running one of
        _in silico_ mutagenesis, variant effect prediction, or prediction,
        prediction/score handlers will accumulate data in memory and only
        write this data to files periodically. By default, Selene will write
        to files when the total amount of data (across all handlers) takes up
        5000MB of space. Please keep in mind that Selene will not monitor the
        memory needed to actually carry out the operations (e.g. variant effect
        prediction) or load the model, so `write_mem_limit` should always be
        less than the total amount of CPU memory you have available on your
        machine. For example, for variant effect prediction, we load all
        the variants in 1 file into memory before getting the predictions, so
        your machine must have enough memory to accommodate that. Another
        possible consideration is your model size and whether you are
        using it on the CPU or a CUDA-enabled GPU (i.e. setting
        `use_cuda` to True).

    Attributes
    ----------
    model : PredMWrapper
        A sequence-based model that has already been trained.
    _seqLen : int
        The length of sequences that the model is expecting.
    batch_size : int
        The size of the mini-batches to use.
    features : list(str)
        The names of the features that the model is predicting.
    use_cuda : bool
        Specifies whether to use a CUDA-enabled GPU or not.
    data_parallel : bool
        Whether to use multiple GPUs or not.
    reference_sequence : class
        The type of sequence on which this analysis will be performed.
    
    '''

    def __init__(self, model, trainedModelPath, features, 
                 analysis, outputDir = None,
                 save_mult_pred = False, outputFormat = 'tsv',
                 seqLen = None, mode = 'prediction', model_built = 'pytorch',
                 batchSize = 64, useCuda = False,
                 dataParallel = False, refSeq = Genome,
                 writeMemLimit = 5000, loggingVerbosity = 2):
        """
        Constructs a new `Analyzer` object.
        """
        # set up the predictive model
        self._model = model
        if not (isinstance(trainedModelPath, str) or \
            hasattr(trainedModelPath, '__len__')):
            raise ValueError(
                '`trainedModelPath` should be a str or list of strs '
                'specifying the full paths to model weights files, but was '
                'type {0}.'.format(type(trainedModelPath)))
        if model._model_built == 'pytorch':
            self._model.initFromFile(trainedModelPath)
            self._model.setMode('evaluate')

            self._dataParallel = dataParallel
            if self._dataParallel:
                self._model.toDataParallel()

            self._useCuda = useCuda
            if self._useCuda:
                self._model.toUseCuda()
        elif model._model_built == 'tensorflow':
            self._model._model = load_model(trainedModelPath)
            # self._model._model.load_weights(trainedModelPath)
            self._model.setMode('evaluate')
            
        self._batchSize = batchSize
        self._features = features
        self._seqLen = seqLen
        if self._seqLen is not None:
            self._startRadius = seqLen // 2
            self._endRadius = self._startRadius
            if seqLen % 2 != 0:
                self._endRadius += 1
        
        # analysis to be done and output handling
        self._analysis = analysis
        self._analysis = set(self._analysis) & set(
            ["diffs", "abs_diffs", "std", "mean_gve", "pval",  "logits", "predictions"])
        self._analysis = sorted(list(self._analysis))
        if len(self._analysis) == 0:
            raise ValueError("'analysis' parameter must be a list that "
                             "contains one of ['diffs', 'abs_diffs', "
                             "'logits', 'predictions'].")
        self._mode = mode
        self._outputDir = outputDir
        self._save_mult_pred = save_mult_pred
        self._outputFormat = outputFormat
        
        self._refSeq = refSeq
        if not self._refSeq._initialized:
            self._refSeq._unpicklable_init()

        self._writeMemLimit = writeMemLimit
        
        initialize_logger(
            os.path.join(self._outputDir, "fugep.log"),
            verbosity = loggingVerbosity)
        

    def _initializeReporters(self, outputPath, colNamesOfIds,
                mult_predictions, save_mult_pred, outputSize, outputFormat = None, analysis = None):
        """
        Initialize the handlers to which FuGEP reports analysis results

        Returns
        -------
        list(fugep.analyze.predict_handlers.PredictionsHandler)
            List of reporters to update as FuGEP receives model predictions.

        """
        if outputFormat is None:
            outputFormat = self._outputFormat
        if analysis is None:
            analysis = self._analysis
        constructor_args = [self._features, colNamesOfIds,
                outputPath, mult_predictions, save_mult_pred,  outputFormat, outputSize,
                self._writeMemLimit // len(analysis)]
        # if self._model._mult_predictions > 1:
        #     self._save_mult_pred = True
        
        reporters = []
        for i, s in enumerate(analysis):
            write_labels = False
            if i == 0:
                write_labels = True
            
            # Handle Parquet-specific handlers first
            if outputFormat == 'parquet':
                if "mean_gve" == s:
                    from .ana_hdl.mean_gve_parquet import MeanGVEParquetHandler
                    reporters.append(MeanGVEParquetHandler(
                        *constructor_args, write_labels=write_labels))
                elif "pval" == s:
                    from .ana_hdl.pval_parquet import PvalParquetHandler
                    reporters.append(PvalParquetHandler(
                        *constructor_args, write_labels=write_labels))
                # For other analysis types with parquet, fall back to existing handlers
                elif "diffs" == s:
                    reporters.append(DiffScoreHandler(
                        *constructor_args, write_labels=write_labels))
                elif "abs_diffs" == s:
                    reporters.append(AbsDiffScoreHandler(
                        *constructor_args, write_labels=write_labels))
                elif "std" == s:
                    reporters.append(StdHandler(
                        *constructor_args, write_labels=write_labels))
                elif "logits" == s:
                    reporters.append(LogitScoreHandler(
                        *constructor_args, write_labels=write_labels))
                elif "predictions" == s and self._mode != "varianteffect":
                    reporters.append(WritePredictionsHandler(
                        *constructor_args, write_labels=write_labels))
                elif "predictions" == s and self._mode == "varianteffect":
                    reporters.append(WriteRefAltHandler(
                        *constructor_args, write_labels=write_labels))
            else:
                # Original logic for TSV/HDF5 formats
                if "diffs" == s:
                    reporters.append(DiffScoreHandler(
                        *constructor_args, write_labels=write_labels))
                elif "abs_diffs" == s:
                    reporters.append(AbsDiffScoreHandler(
                        *constructor_args, write_labels=write_labels))
                elif "std" == s:
                    reporters.append(StdHandler(
                        *constructor_args, write_labels=write_labels))
                elif "mean_gve" == s:
                    reporters.append(MeanGVEHandler(
                        *constructor_args, write_labels=write_labels))
                elif "pval" == s:
                    reporters.append(PvalHandler(
                        *constructor_args, write_labels=write_labels))
                elif "logits" == s:
                    reporters.append(LogitScoreHandler(
                        *constructor_args, write_labels=write_labels))
                elif "predictions" == s and self._mode != "varianteffect":
                    reporters.append(WritePredictionsHandler(
                        *constructor_args, write_labels=write_labels))
                elif "predictions" == s and self._mode == "varianteffect":
                    reporters.append(WriteRefAltHandler(
                        *constructor_args, write_labels=write_labels))
        
        return reporters
        
    def _getSeqFromBedFile(self, bedFilePath,  outputPath,
                   strandIdx = None):
        """
        Get the adjusted sequence coordinates and labels corresponding
        to each row of coordinates in an input BED file. The coordinates
        specified in each row are only used to find the center position
        for the resulting sequence--all regions returned will have the
        length expected by the model.

        Parameters
        ----------
        bedFilePath : str
            Input filepath to BED file.
        outputPath: str
            The path prefix for saving invalid records. Invalid means
            sequences that cannot be fetched, either because
            the exact chromosome cannot be found in the `self._refSeq` FASTA
            file or because the sequence retrieved is out of bounds or overlapping
            with any of the blacklist regions.
        strandIdx : int or None, optional
            Default is None. If sequences must be strand-specific,
            the input BED file may include a column specifying the
            strand ({'+', '-', '.'}).

        Returns
        -------
        list(tup), list(tup)
            The sequence query information (chrom, start, end, strand)
            and the labels (the index, genome coordinates, and sequence
            specified in the BED file).

        """
        sequences = []
        labels = []
        naRows = []
        chrPrefix = True
        for chrom in self._refSeq.get_chrs():
            if not chrom.startswith("chr"):
                chrPrefix = False
                break
        with open(bedFilePath, 'r') as readHdl:
            for i, line in enumerate(readHdl):
                cols = line.strip().split('\t')
                if len(cols) < 3:
                    naRows.append(line)
                    continue
                chrom = cols[0]
                start = cols[1]
                end = cols[2]
                strand = '.'
                if isinstance(strandIdx, int) and len(cols) > strandIdx:
                    strand = cols[strandIdx]
                if 'chr' not in chrom and chrPrefix is True:
                    chrom = "chr{0}".format(chrom)
                elif 'chr' in chrom and chrPrefix is not True:
                    chrom = formatChrom(chrom)
                if not str.isdigit(start) or not str.isdigit(end) \
                        or chrom not in self._refSeq.genome:
                    naRows.append(line)
                    continue
                start, end = int(start), int(end)
                midPos = start + ((end - start) // 2)
                seqStart = midPos - self._startRadius
                seqEnd = midPos + self._endRadius
                if self._refSeq:
                    if not self._refSeq.coords_in_bounds(chrom, seqStart, seqEnd):
                        naRows.append(line)
                        continue
                sequences.append((chrom, seqStart, seqEnd, strand))
                labels.append((i, chrom, start, end, strand))

        if len(naRows) > 0:
            # save NA rows, for which no prediction will be done
            with open(createFilePathWithPrefix(outputPath, 'invalid.bed'), 'w') as fileHdl:
                for naRow in naRows:
                    fileHdl.write(naRow)

        return sequences, labels
    
    def _padOrTruncateSeq(self, sequence):
        if len(sequence) < self._seqLen:
            sequence = _pad_sequence(
                sequence,
                self._seqLen,
                self.reference_sequence.UNK_BASE,
            )
        elif len(sequence) > self._seqLen:
            sequence = _truncate_sequence(sequence, self._seqLen)

        return sequence


