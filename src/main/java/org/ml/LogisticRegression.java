package org.ml;

import jdk.jshell.execution.Util;
import mikera.matrixx.impl.ZeroMatrix;
import mikera.vectorz.Vector;
import mikera.matrixx.Matrix;
import mikera.vectorz.Vector;
import mikera.matrixx.algo.Multiplications;
import mikera.vectorz.Vectorz;
import mikera.vectorz.impl.ZeroVector;

public class LogisticRegression {
    private double learningRate;
    private double regularization;
    private Matrix weights = Matrix.create(Vector.of(1,1,1));

    public void fit(Utils.TrainTestSplitResult tts, int epochs, int batch_size) {
        Matrix X_train = Utils.addBiasColumn(tts.X_train);
        Matrix oneHotYtrain = Utils.oneHot(tts.y_train);

        int samplesCount = X_train.getShape()[0];
        int featuresCount = X_train.getShape()[1];
        int classesCount = oneHotYtrain.getShape()[1];

        weights = Utils.uniformMatrix(featuresCount, classesCount, 0, 0.1);

        for (int epoch = 1; epoch < epochs + 1; epoch++) {
            for (int i = 0; i < samplesCount; i += batch_size) {
                Matrix gradient = Matrix.create(ZeroMatrix.create(featuresCount, classesCount));
                if (i + batch_size < samplesCount) {
                    Matrix y = Multiplications.multiply(X_train, weights);

                    for (int j = i; j < (i+batch_size); j++) {
                        Vector oneHotY = Vector.create(oneHotYtrain.getRow(j));
                        Vector difference = Vector.create(y.getRow(j));
                        difference.sub(oneHotY);

                        Vector xs = Vector.create( X_train.getRow(j));
                        Vector ys = Utils.softmax(difference);

                        Matrix product = Matrix.create(xs.outerProduct(ys));
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
        }
    }
}
