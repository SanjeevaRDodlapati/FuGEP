'''
Created on May 27, 2021

@author: jsun
'''

from abc import ABCMeta
from abc import abstractmethod

from ...data import Genome
from ..analyzer import Analyzer

class SeqAnalyzer(Analyzer, metaclass = ABCMeta):
    '''
    Base class for applying model to performance sequence 
    analysis. This is an abstract class that defines 
    the interface that downstream classes should implement
    '''

    def __init__(self, analysis, mode, model, trainedModelPath,
                 features, model_built= 'pytorch', outputDir = None,
                 save_mult_pred = False, outputFormat = 'tsv',
                 seqLen = None, batchSize = 64, useCuda = False,
                 dataParallel = False, refSeq = Genome,
                 writeMemLimit = 5000, loggingVerbosity = 2, dataTypeConfig = 'memory_optimized'):
        '''
        Construct a new object of 'SeqAnalyzer'
        '''
        super(SeqAnalyzer, self).__init__(model = model, 
                 trainedModelPath = trainedModelPath,
                 features = features, 
                 analysis = analysis, 
                 outputDir = outputDir,
                 save_mult_pred = save_mult_pred,
                 outputFormat = outputFormat,
                 seqLen = seqLen,
                 mode = mode,
                 model_built= model_built,
                 batchSize = batchSize, 
                 useCuda = useCuda,
                 dataParallel = dataParallel, 
                 refSeq = refSeq,
                 writeMemLimit = writeMemLimit,
                 loggingVerbosity = loggingVerbosity,
                 dataTypeConfig = dataTypeConfig)
        
        
    @abstractmethod    
    def evaluate(self, inputData):
        '''
        Evaluate sequence(s) in the inputData, which can be 
        either a dictionary or path to a file.
        The content of the dictionary and the format of the file depends on 
        the type of genomic event to predict
        '''
        
        raise NotImplementedError()