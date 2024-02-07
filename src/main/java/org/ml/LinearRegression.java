package org.ml;
import mikera.matrixx.Matrix;
import mikera.vectorz.Vector;

import java.util.Random;

public class LinearRegression<T extends Number> {
    private float learning_rate;
    private float regularization;
    private Vector weights;

    public LinearRegression(float learning_rate, float regularization) {
        this.learning_rate = learning_rate;
        this.regularization = regularization;
    }

    public void fit(Matrix data, Vector targets, int epochs, int batch_size) throws Exception {
        Matrix data_with_bias = Utils.addBiasColumn(data);

        int number_of_samples = data_with_bias.rowCount();
        int number_of_features = data_with_bias.columnCount();

        weights = Utils.uniformVector(number_of_features, 0, 0.1);

        for (int ep = 0; ep < epochs; ep++) {
            for (int i = 0; i < (number_of_samples - (number_of_samples % batch_size)); i += batch_size) {
                Vector gradient = Vector.create(number_of_features);
                gradient.fill(0);
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
            Matrix matrix_weights = Matrix.create(weights).toMatrixTranspose();

            Matrix y_pred = (Matrix) data_with_bias.multiplyCopy(matrix_weights);
            y_pred = y_pred.toMatrixTranspose();
            Vector vectorized_preds = Vector.create(y_pred.getRow(0));

            System.out.println("Epoch: " + (ep + 1) + ", RMSE: " + Utils.rmse(y_true, vectorized_preds));
        }
    }

    public double predict(Vector v) {
        Matrix new_mat = Matrix.create(v);
        Matrix to_predict = Utils.addBiasColumn(new_mat);
        if (to_predict.columnCount() != weights.length())
            throw new IllegalArgumentException("Number of features differs from the train set");
        return to_predict.multiplyCopy(weights).get(0,0);
    }

    public Vector predict(Matrix M) {
        Matrix to_predict = Utils.addBiasColumn(M);
        to_predict.multiply(weights);
        Vector res = Vector.create(to_predict.toMatrixTranspose().getRow(0));;
        return res;
    }
}
