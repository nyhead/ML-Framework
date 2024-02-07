package org.ml;

import mikera.matrixx.Matrix;
import mikera.vectorz.Vector;

import java.util.ArrayList;
import java.util.Arrays;

import static java.lang.Math.*;

import java.util.List;
import java.util.Random;
public class Utils {
    static double rmse(Vector y_true, Vector y_pred) throws Exception {
        if (!y_true.isSameShape(y_pred)) {
            throw new IllegalArgumentException("Vectors must be of the same length");
        }

        double sum = 0;
        for (int i = 0; i < y_true.length(); i++) {
            double diff = y_true.get(i) - y_pred.get(i);
            sum += pow(diff, 2);
        }
        double r = 0;
        r = sum / y_true.length();
        return sqrt(r);
    }

    static double sigmoid(double x) {
        return (1 / (1 + exp(x)));
    }
    static Matrix oneHot(Vector targets) {
        double[] target_vals = targets.asDoubleArray();
        int max = (int) Arrays.stream(target_vals).max().orElse(0.0);
        Vector v  = Vector.createLength(max + 1);
        v.fill(0);
        Matrix res_mat = Matrix.wrap(targets.length(),1, v.asDoubleArray());
        for (int i = 0; i < targets.length(); i++) {
            res_mat.set(i, (int) target_vals[i], 1);
        }
        return res_mat;
    }

    public static Matrix addBiasColumn(Matrix M) {
        Matrix mat = M.copy(); // Get the original matrix data

        // Iterate over each row of the matrix
        for (var row : mat) {
            // Assuming T can be an Integer, we add 1 as the bias.
            // This requires casting since our matrix is generic.
            // This is a limitation in Java compared to C++ templates.
            row.add(1.0);
        }

        // Create a new Matrix with the modified matrix data
        Matrix res = new Matrix(mat);
        return res;
    }
    public static Matrix uniformMatrix(int height, int width, double mean, double std) {
        Random random = new Random();
        Matrix matrix = new Matrix(height, width);

        for (int i = 0; i < height; i++) {
            List<Double> row = new ArrayList<>();
            for (int j = 0; j < width; j++) {
                double value = mean + std * random.nextGaussian(); // Generate normally distributed value
                row.add(value);
            }
            matrix.setRow(i, (Vector.create(row)));
        }

        return matrix;
    }
    public static Vector uniformVector(int length, double mean, double std) {
        Random random = new Random();
        List<Double> vector = new ArrayList<>();

        for (int i = 0; i < length; i++) {
            // Generate normally distributed value with specified mean and std
            double value = mean + std * random.nextGaussian();
            vector.add(value);
        }

        return Vector.create(vector);
    }
//    static Matrix fromOneHot()
}
