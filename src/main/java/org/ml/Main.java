package org.ml;

import mikera.vectorz.ops.Log;
import org.ml.util.Util;

import java.io.*;
import java.util.*;

public class Main {
    public static void main(String[] args) throws Exception {
        String csvFile = "D:/winter-2023/java_related/ML-Framework/generated_data.csv";
        BufferedReader br = new BufferedReader(new FileReader(csvFile));
        String line;
        ArrayList<Double> targets = new ArrayList<>();
        ArrayList<ArrayList<Double>> data = new ArrayList<>();

        while ((line = br.readLine()) != null) {
            String[] values = line.split(",");
            ArrayList<Double> trainRow = new ArrayList<>();
            // Assuming the last column is the target
            for (int i = 0; i < values.length - 1; i++) {
                trainRow.add(Double.valueOf(values[i]));
            }
            targets.add(Double.parseDouble(values[values.length -  1]));

            data.add(trainRow);
        }

        br.close();
//        targets.forEach(System.out::println);
//        data.forEach(System.out::println);
        double[][] arrayData = data.stream()
                .map(l -> l.stream().mapToDouble(Double::doubleValue).toArray())
                .toArray(double[][]::new);
        double[] arrayTargets = targets.stream().mapToDouble(d -> d).toArray();

        int seed = 42;
        Util.TrainTestSplitResult train_test_split = Util.trainTestSplit(arrayData, arrayTargets, .85, seed);
//        Matrix XT = train_data.toMatrixTranspose();
//        Matrix XTX = Matrix.create(Multiplications.multiply(XT, train_data).inverse());
//        Matrix res = Multiplications.multiply(Multiplications.multiply(XTX, XT), Matrix.create(train_target));
//
//        System.out.println("RMSE: " + Utils.rmse(y_true, vectorized_preds));
        LinearRegression model = new LinearRegression(0.01,0.5, seed);
        model.fit(train_test_split, 9,10);
        }
    }

