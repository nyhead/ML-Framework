package org.ml;

import mikera.matrixx.impl.ZeroMatrix;
import mikera.vectorz.GrowableVector;
import mikera.vectorz.Vector;
import mikera.matrixx.Matrix;
import mikera.matrixx.algo.Multiplications;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.ml.util.Util;

import java.util.ArrayList;
import java.util.List;


public class LogisticRegression {
    private double learningRate;
    private double regularization;
    private Matrix weights;
    public int seed = 42;

    public LogisticRegression(double learningRate, double regularization, int seed) {
        this.learningRate = learningRate;
        this.regularization = regularization;
        this.seed = seed;
    }

    public void fit(Util.TrainTestSplitResult tts, int epochs, int batch_size) {
        Matrix X_train = Util.addBiasColumn(tts.X_train);
        Matrix oneHotYtrain = Util.oneHot(tts.y_train);

        int samplesCount = X_train.getShape()[0];
        int featuresCount = X_train.getShape()[1];
        int classesCount = oneHotYtrain.getShape()[1];

        weights = Util.uniformMatrix(featuresCount, classesCount, 0, 0.1, seed);
        List<Double> lossValues = new ArrayList<>();

        for (int epoch = 1; epoch < epochs + 1; epoch++) {
            for (int i = 0; i < samplesCount; i += batch_size) {
                Matrix gradient = ZeroMatrix.create(featuresCount, classesCount).toMatrix();
                if (i + batch_size < samplesCount) {
                    Matrix y = Multiplications.multiply(X_train, weights);

                    for (int j = i; j < (i+batch_size); j++) {
                        Vector oneHotY = Vector.create(oneHotYtrain.getRow(j));
                        Vector difference = y.getRow(j).toVector();
                        difference.sub(oneHotY);

                        Vector xs = X_train.getRow(j).toVector();
                        Vector ys = Util.softmax(difference);

                        Matrix product = xs.outerProduct(ys).toMatrix();
                        product.divide(batch_size);
                        gradient.add(product);
                    }
                }

                gradient.multiply(learningRate);
                weights.sub(gradient);
            }
            Matrix probabilities = predictProbs(tts.X_train);
            Vector yPred = predict(tts.X_train);
            Vector yTrue = tts.y_train;
            double loss = Util.crossEntropy(oneHotYtrain, probabilities);
            lossValues.add(loss);
            System.out.println("Epoch: " + epoch + ", Loss(Cross-Entropy): "
                    + Util.crossEntropy(oneHotYtrain, probabilities) + Util.accuracy(yTrue, yPred));
        }
        plotLoss(lossValues, epochs);

    }
    private void plotLoss(List<Double> lossValues, int epochs) {
        // Create a chart
        XYChart chart = new XYChart(600,  400);
        chart.setTitle("Loss over Epochs");
        chart.setXAxisTitle("Epochs");
        chart.setYAxisTitle("Loss (Cross-Entropy)");

        // Add loss series
        double[] xData = new double[epochs];
        for (int i =  0; i < epochs; i++) {
            xData[i] = i +  1; // Epoch numbers
        }
        double[] yData = lossValues.stream().mapToDouble(Double::doubleValue).toArray();
        chart.addSeries("Loss", xData, yData);

        // Show the chart
        new SwingWrapper<>(chart).displayChart();
    }
    private Matrix predictProbs(Matrix unbiased) {
        Matrix biasedX = Util.addBiasColumn((unbiased));
        return Util.softmax(Multiplications.multiply(biasedX, weights));
    }

    private Vector predict(Matrix unbiased) {
        GrowableVector classes = new GrowableVector();
        Matrix probabilities = this.predictProbs(unbiased);

        for (int i = 0; i < probabilities.getShape()[0]; i++) {
            Vector v = probabilities.getRow(i).toVector();
            classes.append(v.maxElementIndex());
        }

        return classes.toVector();
    }
}
