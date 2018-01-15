package org.deeplearning4j.examples.dataexamples;


import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ResizeImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This code example is featured in this youtube video
 * https://www.youtube.com/watch?v=ECA6y6ahH5E
 *
 * This differs slightly from the Video Example,
 * The Video example had the data already downloaded
 * This example includes code that downloads the data
 *
 * Data is downloaded from
 * wget http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
 * followed by tar xzvf mnist_png.tar.gz
 *
 * This examples builds on the MnistImagePipelineExample
 * by adding a Neural Net
 */
public class MnistImagePipelineExampleAddNeuralNet {
  private static Logger log = LoggerFactory.getLogger(MnistImagePipelineExampleAddNeuralNet.class);

  /** Data URL for downloading */
  public static final String DATA_URL = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";

  /** Location to save and extract the training/testing data */
  public static final String DATA_PATH = FilenameUtils.concat("E:\\ML\\DL", "dl4j_Mnist/");

  public static void main(String[] args) throws Exception {
    // image information
    // 28 * 28 grayscale
    // grayscale implies single channel
    int height = 32;
    int width = 30;
    int channels = 1;
    int rngseed = 123;
    Random randNumGen = new Random(rngseed);
    int batchSize = 128;
    int outputNumTrain = 20;
    int numEpochs = 50;

    /*
    This class downloadData() downloads the data
    stores the data in java's tmpdir 15MB download compressed
    It will take 158MB of space when uncompressed
    The data can be downloaded manually here
    http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
    */
    MnistImagePipelineExample.downloadData();

    // Define the File Paths
    File trainData = new File(MnistImagePipelineExampleAddNeuralNet.class.getClassLoader().getResource("faces_4").getFile(),"training");
//    File trainData = new File(DATA_PATH + "/mnist_png/training");
    File testData = new File(MnistImagePipelineExampleAddNeuralNet.class.getClassLoader().getResource("faces_4").getFile(),"testing");

    // Define the FileSplit(PATH, ALLOWED FORMATS,random)
    FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
    FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

    // Extract the parent path as the image label
    ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    
    ResizeImageTransform resize = new ResizeImageTransform(width, height);
    ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker,resize);

    // Initialize the record reader
    // add a listener, to extract the name
    recordReader.initialize(train);
    //recordReader.setListeners(new LogRecordListener());

    // DataSet Iterator
    DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNumTrain);

    // Scale pixel values to 0-1
    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    scaler.fit(dataIter);
    dataIter.setPreProcessor(scaler);

    // Build Our Neural Network
    log.info("BUILD MODEL");
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(rngseed)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .iterations(1)
        .learningRate(0.006)
        .updater(Updater.NESTEROVS)
        .regularization(true).l2(1e-4)
        .list()
        .layer(0, new DenseLayer.Builder()
            .nIn(height * width)
            .nOut(100)
            .activation(Activation.RELU)
            .weightInit(WeightInit.XAVIER)
            .build())
        .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nIn(100)
            .nOut(outputNumTrain)
            .activation(Activation.SOFTMAX)
            .weightInit(WeightInit.XAVIER)
            .build())
        .pretrain(false).backprop(true)
        .setInputType(InputType.convolutional(height, width, channels))
        .build();

    MultiLayerNetwork model = new MultiLayerNetwork(conf);

    // The Score iteration Listener will log
    // output to show how well the network is training
    model.setListeners(new ScoreIterationListener(10));

    log.info("TRAIN MODEL");
    for (int i = 0; i < numEpochs; i++) {
      model.fit(dataIter);
    }
    recordReader.close();
    log.info("EVALUATE MODEL");
    ImageRecordReader recordReaderTest = new ImageRecordReader(height, width, channels, labelMaker,resize);


    // The model trained on the training dataset split
    // now that it has trained we evaluate against the
    // test data of images the network has not seen

    recordReaderTest.initialize(test);
    NativeImageLoader loader = new NativeImageLoader(height, width, channels);
    Evaluation eval = new Evaluation(20);
    System.out.println(dataIter.getLabels());
    Map<String,Integer> labels = new HashMap<String,Integer>();
    for(int i=0; i<dataIter.getLabels().size();i++) {
    	labels.put(dataIter.getLabels().get(i), i);
    }
    while(recordReaderTest.hasNext()) {
    	recordReaderTest.next();
    	File currentFile = recordReaderTest.getCurrentFile();
		INDArray asMatrix = loader.asMatrix(currentFile);
    	String substring = currentFile.getName().substring(0, currentFile.getName().indexOf("_"));
		int actualIdx = labels.get(substring);
		INDArray output = model.output(asMatrix);
		for(int i=0; i<dataIter.getLabels().size(); i++) {
			if(output.getDouble(i)==1.00) {
				eval.eval(i, actualIdx); break;
			}
		}
		

    }
    System.out.println(eval.stats());
    recordReaderTest.close();
 }

}
