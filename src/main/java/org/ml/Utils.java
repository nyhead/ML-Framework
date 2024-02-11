package org.ml;

import mikera.matrixx.Matrix;
import mikera.vectorz.Vector;

import java.util.ArrayList;
import java.util.Arrays;

import static java.lang.Math.*;

import java.util.List;
import java.util.Random;
public class Utils {
    public static class TrainTestSplitResult {
        public Matrix X_train;
        public Matrix X_test;
        public Vector y_train;
        public Vector y_test;

        public TrainTestSplitResult(Matrix X_train, Matrix X_test, Vector y_train, Vector y_test) {
            this.X_train = X_train;
            this.X_test = X_test;
            this.y_train = y_train;
            this.y_test = y_test;
        }
    }
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
        random.setSeed(92);
        List<Double> vector = new ArrayList<>();

        for (int i = 0; i < length; i++) {
            // Generate normally distributed value with specified mean and std
            double value = mean + std * random.nextGaussian();
            vector.add(value);
        }

        return Vector.create(vector);
    }
    public static TrainTestSplitResult trainTestSplit(double[][] X, double[] y, double testSize, long seed) {
        int totalSize = X.length;
        int testSizeCount = (int) (totalSize * testSize);
        int trainSizeCount = totalSize - testSizeCount;

        // Initialize arrays for the split
        double[][] X_train = new double[trainSizeCount][];
        double[][] X_test = new double[testSizeCount][];
        double[] y_train = new double[trainSizeCount];
        double[] y_test = new double[testSizeCount];

        // Create an array of indices and shuffle it
        int[] indices = new int[totalSize];
        for (int i = 0; i < totalSize; i++) {
            indices[i] = i;
        }
        shuffleArray(indices, seed);

        // Split the data based on the shuffled indices
        for (int i = 0; i < trainSizeCount; i++) {
            X_train[i] = X[indices[i]];
            y_train[i] = y[indices[i]];
        }
        for (int i = 0; i < testSizeCount; i++) {
            X_test[i] = X[indices[trainSizeCount + i]];
            y_test[i] = y[indices[trainSizeCount + i]];
        }

        return new TrainTestSplitResult(Matrix.create(X_train), Matrix.create(X_test), Vector.create(y_train), Vector.create(y_test));
    }

    private static void shuffleArray(int[] array, long seed) {
        Random rand = new Random(seed);
        for (int i = array.length - 1; i > 0; i--) {
            int index = rand.nextInt(i + 1);
            // Simple swap
            int a = array[index];
            array[index] = array[i];
            array[i] = a;
        }
    }
//    static Matrix fromOneHot()
}
