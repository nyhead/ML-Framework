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

    public void fit(Matrix data, Vector targets, int epochs, int batch_size) throws Exception {
        Matrix data_with_bias = Utils.addBiasColumn(data);

        int number_of_samples = data_with_bias.rowCount();
        int number_of_features = data_with_bias.columnCount();

        weights = Vector.create(Vectorz.createUniformRandomVector(number_of_features));

        for (int ep = 0; ep < epochs; ep++) {
            for (int i = 0; i < (number_of_samples - (number_of_samples % batch_size)); i += batch_size) {
                Vector gradient = Vector.create(ZeroVector.create(number_of_features));
                for (int j = i; j < i + batch_size; j++) {
                    Vector sample = Vector.create(data_with_bias.getRow(j));
                    double target = targets.get(j);

                    double predicted = sample.innerProduct(weights).get();

                    predicted = predicted - target;

                    sample.multiply(predicted);
                    Vector grad = (Vector) sample.divideCopy(batch_size);

                    gradient.add(grad);
                }
                Vector reg_weights = weights.multiplyCopy(regularization);
                gradient.add(reg_weights);
                Vector weights_update = gradient.multiplyCopy(learning_rate);
                weights.sub(weights_update);
            }
            Vector y_true = targets;

            Matrix matrix_weights = Matrix.create(new double[][]{weights.toDoubleArray()}).toMatrixTranspose();

            Matrix y_pred = Multiplications.multiply(data_with_bias, matrix_weights);
            y_pred = y_pred.toMatrixTranspose();
            Vector vectorized_preds = Vector.create(y_pred.getRow(0));
            System.out.println(weights);
            System.out.println("Epoch: " + (ep + 1) + ", RMSE: " + Utils.rmse(y_true, vectorized_preds));
        }
    }

    public Vector predict(Matrix M) {
        Matrix to_predict = Utils.addBiasColumn(M);
        Matrix matrix_weights = Matrix.create(new double[][]{weights.toDoubleArray()}).toMatrixTranspose();

        Matrix y_pred = Multiplications.multiply(to_predict, matrix_weights);
        Vector vectorized_preds = Vector.create(y_pred.toMatrixTranspose().getRow(0));
        return vectorized_preds;
    }
}
