package org.ml;
import mikera.matrixx.Matrix;
import mikera.vectorz.Vector;
import mikera.matrixx.algo.Multiplications;
import mikera.vectorz.Vectorz;
import mikera.vectorz.impl.ZeroVector;

import java.util.Random;

public class LinearRegression {
    private double learning_rate;
    private double regularization;
    private Vector weights;

    public LinearRegression(double learning_rate, double regularization) {
        this.learning_rate = learning_rate;
        this.regularization = regularization;
    }

    public void fit(Utils.TrainTestSplitResult tts, int epochs, int batch_size) throws Exception {
        Vector y_train = tts.y_train;
        Matrix X_train = Utils.addBiasColumn(tts.X_train);

        int number_of_samples = X_train.rowCount();
        int number_of_features = X_train.columnCount();

        weights = Vector.create(Vectorz.createUniformRandomVector(number_of_features));

        for (int ep = 0; ep < epochs; ep++) {
            for (int i = 0; i < (number_of_samples - (number_of_samples % batch_size)); i += batch_size) {
                Vector gradient = Vector.create(ZeroVector.create(number_of_features));
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
                Vector weights_update = gradient.multiplyCopy(learning_rate);
                weights.sub(weights_update);
            }

            Vector train_predict = predict(X_train, false);
            Vector test_predict = predict(tts.X_test, true);
            System.out.println(weights);
            System.out.println("Epoch: " + (ep + 1) + ", Train RMSE: " + Utils.rmse(y_train, train_predict) + ", Test RMSE: " + Utils.rmse(tts.y_test, test_predict));
        }
    }

    public Vector predict(Matrix M, boolean bias) {
        Matrix to_predict = bias ? Utils.addBiasColumn(M) : M;
        Matrix matrix_weights = Matrix.create(new double[][]{weights.toDoubleArray()}).toMatrixTranspose();

        Matrix y_pred = Multiplications.multiply(to_predict, matrix_weights);
        Vector vectorized_preds = Vector.create(y_pred.toMatrixTranspose().getRow(0));
        return vectorized_preds;
    }
}
