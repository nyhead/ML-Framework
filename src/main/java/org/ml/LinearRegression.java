package org.ml;
import mikera.matrixx.Matrix;
import mikera.matrixx.Matrixx;
import mikera.vectorz.Vector;
import mikera.matrixx.algo.Multiplications;
import mikera.vectorz.Vectorz;
import mikera.vectorz.impl.ZeroVector;
import org.ml.util.Util;

public class LinearRegression {
    private double learningRate;
    private double regularization;
    private Vector weights = Vector.of(1,0);
    public int seed = 42;

    public LinearRegression(double learningRate, double regularization, int seed) {
        this.learningRate = learningRate;
        this.regularization = regularization;
        this.seed = seed;
    }

    public void fit(Util.TrainTestSplitResult tts, int epochs, int batch_size) throws Exception {
        Vector y_train = tts.y_train;
        Matrix X_train = Util.addBiasColumn(tts.X_train);

        int samplesCount = X_train.rowCount();
        int featuresCount = X_train.columnCount();

        weights = Util.uniformVector(featuresCount, 0, 0.1, seed);
        for (int ep = 0; ep < epochs; ep++) {
            for (int i = 0; i < (samplesCount - (samplesCount % batch_size)); i += batch_size) {
                Vector gradient = Vector.create(ZeroVector.create(featuresCount));
                for (int j = i; j < i + batch_size; j++) {
                    Vector sample = Vector.create(X_train.getRow(j));
                    double target = y_train.get(j);

                    double predicted = sample.innerProduct(weights).get();

                    predicted = predicted - target;

                    sample.multiply(predicted);
                    Vector grad = sample.divideCopy(batch_size).toVector();

                    gradient.add(grad);
                }
                Vector reg_weights = weights.multiplyCopy(regularization);
                gradient.add(reg_weights);
                Vector weights_update = gradient.multiplyCopy(learningRate);
                weights.sub(weights_update);
            }

            Vector train_predict = predict(X_train, false);
            Vector test_predict = predict(tts.X_test, true);
            System.out.println(weights);
            System.out.println("Epoch: " + (ep + 1) + ", Train RMSE: " + Util.rmse(y_train, train_predict) + ", Test RMSE: " + Util.rmse(tts.y_test, test_predict));
        }
    }

    public Vector predict(Matrix M, boolean bias) {
        Matrix to_predict = bias ? Util.addBiasColumn(M) : M;
        Matrix matrix_weights = Matrix.create(new double[][]{weights.toDoubleArray()}).toMatrixTranspose();

        Matrix yPred = Multiplications.multiply(to_predict, matrix_weights);
        Vector vectorized_preds = Vector.create(yPred.toMatrixTranspose().getRow(0));
        return vectorized_preds;
    }
}
