package org.ml.util;

import mikera.matrixx.Matrix;
import mikera.vectorz.Vector;
import mikera.vectorz.impl.ArraySubVector;
import mikera.vectorz.impl.GrowableIndexedVector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static java.lang.Math.*;

public class Util {
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

    public static double rmse(Vector y_true, Vector y_pred) throws Exception {
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

    public static Vector softmax(Vector v) {
        double max = v.elementMax();
        Vector maxSubV = v.clone();
        maxSubV.sub(max);
        maxSubV.exp();
        return maxSubV;
    }

    public static Matrix softmax(Matrix m) {
        Matrix result = Matrix.create(m.rowCount(), m.columnCount());

        for (int i = 0; i < m.rowCount(); i++) {
            Vector row = m.getRow(i).toVector();
            double max = row.elementMax();

            for (int j = 0; j < row.length(); j++) {
                row.set(j, row.get(j) - max);
            }

            row.exp();

            double sum = row.elementSum();
            for (int j = 0; j < row.length(); j++) {
                row.set(j, row.get(j) / sum);
            }

            result.setRow(i, row);
        }

        return result;
    }

public static Matrix oneHot(Vector targets) {
    int numClasses = (int) targets.elementMax() + 1;
    Matrix result = Matrix.create(targets.length(), numClasses);

    for (int i = 0; i < targets.length(); i++) {
        int target = (int) targets.get(i);
        result.set(i, target, 1);
    }

    return result;
}
    public static Matrix addBiasColumn(Matrix M) {
        var shape = M.getShape();
        shape[1] += 1;
        Matrix mat = Matrix.create(shape);
        for (int i = 0; i < mat.rowCount(); i++) {
            ArraySubVector row = M.getRow(i);
            GrowableIndexedVector newRow = GrowableIndexedVector.create(row);
            newRow.append(1.0);
            mat.setRow(i, newRow);
        }

        return mat;
    }

    public static Matrix uniformMatrix(int height, int width, double mean, double std, int seed) {
        Random random = new Random();
        random.setSeed(seed);
        Matrix matrix = new Matrix(height, width);

        for (int i = 0; i < height; i++) {
            List<Double> row = new ArrayList<>();
            for (int j = 0; j < width; j++) {
                // Generate normally distributed value
                double value = mean + std * random.nextGaussian();
                row.add(value);
            }
            matrix.setRow(i, (Vector.create(row)));
        }

        return matrix;
    }

    public static Vector uniformVector(int length, double mean, double std, int seed) {
        Random random = new Random();
        random.setSeed(seed);
        List<Double> vector = new ArrayList<>();

        for (int i = 0; i < length; i++) {
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
            int a = array[index];
            array[index] = array[i];
            array[i] = a;
        }
    }

    public static double accuracy(Vector yTrue, Matrix yPred) {
        //converts one-hot encoded vectors to class indices
        Vector flattenedPreds = fromOneHot(yPred);
        return accuracy(yTrue, flattenedPreds);
    }

    public static double accuracy(Vector yTrue, Vector yPred) {
        if (yTrue.length() != yPred.length()) {
            throw new IllegalArgumentException("Vectors must be of the same length");
        }

        double counter = 0;
        for (int i = 0; i < yTrue.length(); i++) {
            if (yTrue.get(i) == yPred.get(i)) {
                counter++;
            }
        }

        return counter / yTrue.length();
    }

    public static double crossEntropy(Vector yTrue, Vector yPred) {
        double loss = 0.0;
        for (int i = 0; i < yPred.length(); i++) {
            // Check to prevent log(0)
            if (yPred.get(i) > 0) {
                loss += yTrue.get(i) * Math.log(yPred.get(i));
            }
        }
        return -loss;
    }

    public static double crossEntropy(Matrix yTrue, Matrix yPred) {
        double loss = 0.0;
        int batchSize = yTrue.rowCount();
        int numClasses = yTrue.columnCount();

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < numClasses; j++) {
                double trueVal = yTrue.get(i, j);
                double predVal = yPred.get(i, j);
                if (predVal > 0) {
                    loss += trueVal * Math.log(predVal);
                }
            }
        }
        return -loss / batchSize;
    }

    public static Vector fromOneHot(Matrix m) {
        int rows = m.rowCount();
        double[] resVec = new double[rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < m.columnCount(); j++) {
                if (m.get(i, j) == 1.0) {
                    resVec[i] = j;
                    break;
                }
            }
        }

        return Vector.of(resVec);
    }
}
